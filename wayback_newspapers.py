"""
Wayback Machine Newspaper Downloader
====================================

A toolkit for searching and downloading historical newspapers from:
1. Wayback Machine (archived newspaper websites)
2. Internet Archive Newspaper Collections

APIs used:
- Wayback CDX Server API: https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
- Internet Archive Search/Scrape API
- Internet Archive Download API

Usage:
    python wayback_newspapers.py

For bulk downloads, set your Internet Archive credentials:
    export IA_ACCESS_KEY="your_access_key"
    export IA_SECRET_KEY="your_secret_key"

Or create a config file at ~/.ia with:
    [s3]
    access = your_access_key
    secret = your_secret_key
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import quote
import hashlib

# Optional: HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Output directory
OUTPUT_DIR = Path(__file__).parent / "wayback-output"


def get_output_path(filename: str) -> Path:
    """Get path in output directory, creating it if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


class WaybackNewspaperDownloader:
    """Download historical newspapers from Wayback Machine and Internet Archive."""

    # API endpoints
    CDX_API = "https://web.archive.org/cdx/search/cdx"
    AVAILABILITY_API = "https://archive.org/wayback/available"
    WAYBACK_BASE = "https://web.archive.org/web"
    IA_SEARCH_API = "https://archive.org/advancedsearch.php"
    IA_SCRAPE_API = "https://archive.org/services/search/v1/scrape"
    IA_METADATA_API = "https://archive.org/metadata"
    IA_DOWNLOAD_BASE = "https://archive.org/download"

    # Known newspaper website patterns
    NEWSPAPER_DOMAINS = [
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "latimes.com",
        "chicagotribune.com",
        "bostonglobe.com",
        "sfchronicle.com",
        "usatoday.com",
        "theguardian.com",
        "telegraph.co.uk",
        "bbc.co.uk/news",
        "cnn.com",
        "reuters.com",
        "apnews.com",
    ]

    # Internet Archive newspaper collections
    IA_NEWSPAPER_COLLECTIONS = [
        "newspapers",
        "americana",
        "historicalnewspapers",
        "kentuckynewspapers",
        "tennessean",
        "washingtontimes",
    ]

    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            access_key: Internet Archive S3-like access key (optional)
            secret_key: Internet Archive S3-like secret key (optional)
        """
        self.access_key = access_key or os.environ.get("IA_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("IA_SECRET_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WaybackNewspaperDownloader/1.0 (Research purposes)"
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None,
                      stream: bool = False) -> requests.Response:
        """Make a rate-limited HTTP request."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, stream=stream, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            raise

    # ==================== WAYBACK CDX API ====================

    def search_wayback(self, url: str, from_date: Optional[str] = None,
                       to_date: Optional[str] = None, limit: int = 100,
                       match_type: str = "prefix") -> List[Dict]:
        """
        Search Wayback Machine for archived snapshots of a URL.

        Args:
            url: Website URL to search (e.g., "nytimes.com")
            from_date: Start date (YYYYMMDD format)
            to_date: End date (YYYYMMDD format)
            limit: Maximum results to return
            match_type: "exact", "prefix", "host", or "domain"

        Returns:
            List of snapshot dictionaries
        """
        print(f"\n{'='*60}")
        print(f"SEARCHING WAYBACK MACHINE: {url}")
        print(f"{'='*60}")

        params = {
            "url": url,
            "output": "json",
            "matchType": match_type,
            "limit": limit,
            "fl": "timestamp,original,mimetype,statuscode,digest,length"
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            response = self._make_request(self.CDX_API, params=params)
            data = response.json()

            if not data:
                print("  No results found")
                return []

            # First row is headers
            headers = data[0]
            results = []

            for row in data[1:]:
                snapshot = dict(zip(headers, row))
                snapshot["wayback_url"] = f"{self.WAYBACK_BASE}/{snapshot['timestamp']}/{snapshot['original']}"
                results.append(snapshot)

            print(f"  Found {len(results)} snapshots")
            if results:
                first_date = results[0]["timestamp"][:8]
                last_date = results[-1]["timestamp"][:8]
                print(f"  Date range: {first_date} - {last_date}")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return []

    def check_availability(self, url: str, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Check if a URL has an available snapshot.

        Args:
            url: URL to check
            timestamp: Optional specific timestamp (YYYYMMDD)

        Returns:
            Snapshot info or None
        """
        params = {"url": url}
        if timestamp:
            params["timestamp"] = timestamp

        try:
            response = self._make_request(self.AVAILABILITY_API, params=params)
            data = response.json()

            if data.get("archived_snapshots", {}).get("closest"):
                return data["archived_snapshots"]["closest"]
            return None

        except Exception as e:
            print(f"  Availability check failed: {e}")
            return None

    def download_wayback_page(self, wayback_url: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Download a page from Wayback Machine.

        Args:
            wayback_url: Full Wayback URL (e.g., https://web.archive.org/web/20200101/...)
            output_path: Where to save the file

        Returns:
            Path to downloaded file or None
        """
        if output_path is None:
            # Generate filename from URL
            url_hash = hashlib.md5(wayback_url.encode()).hexdigest()[:8]
            timestamp = wayback_url.split("/web/")[1].split("/")[0] if "/web/" in wayback_url else "unknown"
            output_path = get_output_path(f"wayback_{timestamp}_{url_hash}.html")

        try:
            print(f"  Downloading: {wayback_url[:80]}...")
            response = self._make_request(wayback_url, stream=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  Saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    # ==================== INTERNET ARCHIVE COLLECTIONS ====================

    def search_ia_newspapers(self, query: str, collection: Optional[str] = None,
                             rows: int = 50) -> List[Dict]:
        """
        Search Internet Archive for newspaper items.

        Args:
            query: Search query
            collection: Specific collection to search (e.g., "newspapers")
            rows: Number of results

        Returns:
            List of item dictionaries
        """
        print(f"\n{'='*60}")
        print(f"SEARCHING INTERNET ARCHIVE: {query}")
        print(f"{'='*60}")

        # Build query
        q_parts = [query]
        if collection:
            q_parts.append(f"collection:{collection}")
        else:
            # Search across newspaper collections
            collection_query = " OR ".join([f"collection:{c}" for c in self.IA_NEWSPAPER_COLLECTIONS])
            q_parts.append(f"({collection_query})")

        params = {
            "q": " AND ".join(q_parts),
            "fl[]": ["identifier", "title", "date", "creator", "collection", "mediatype", "item_size"],
            "sort[]": "date desc",
            "rows": rows,
            "page": 1,
            "output": "json"
        }

        try:
            response = self._make_request(self.IA_SEARCH_API, params=params)
            data = response.json()

            results = data.get("response", {}).get("docs", [])
            total = data.get("response", {}).get("numFound", 0)

            print(f"  Found {total} total items, returning {len(results)}")

            for item in results[:5]:
                print(f"    - {item.get('identifier')}: {item.get('title', 'No title')[:50]}")

            return results

        except Exception as e:
            print(f"  Error: {e}")
            return []

    def scrape_collection(self, collection: str, fields: Optional[List[str]] = None,
                          count: int = 100) -> List[Dict]:
        """
        Scrape metadata from an Internet Archive collection.

        Args:
            collection: Collection identifier
            fields: Fields to retrieve
            count: Number of items

        Returns:
            List of item dictionaries
        """
        print(f"\n{'='*60}")
        print(f"SCRAPING COLLECTION: {collection}")
        print(f"{'='*60}")

        if fields is None:
            fields = ["identifier", "title", "date", "creator", "item_size"]

        params = {
            "q": f"collection:{collection}",
            "fields": ",".join(fields),
            "count": count
        }

        try:
            response = self._make_request(self.IA_SCRAPE_API, params=params)
            data = response.json()

            items = data.get("items", [])
            total = data.get("total", 0)

            print(f"  Total in collection: {total}")
            print(f"  Retrieved: {len(items)}")

            return items

        except Exception as e:
            print(f"  Error: {e}")
            return []

    def get_item_metadata(self, identifier: str) -> Optional[Dict]:
        """
        Get full metadata for an Internet Archive item.

        Args:
            identifier: Item identifier

        Returns:
            Metadata dictionary or None
        """
        url = f"{self.IA_METADATA_API}/{identifier}"

        try:
            response = self._make_request(url)
            return response.json()

        except Exception as e:
            print(f"  Metadata fetch failed: {e}")
            return None

    def list_item_files(self, identifier: str) -> List[Dict]:
        """
        List all files in an Internet Archive item.

        Args:
            identifier: Item identifier

        Returns:
            List of file dictionaries
        """
        metadata = self.get_item_metadata(identifier)
        if metadata:
            return metadata.get("files", [])
        return []

    def download_ia_file(self, identifier: str, filename: str,
                         output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Download a file from Internet Archive.

        Args:
            identifier: Item identifier
            filename: File name within the item
            output_path: Where to save

        Returns:
            Path to downloaded file or None
        """
        url = f"{self.IA_DOWNLOAD_BASE}/{identifier}/{filename}"

        if output_path is None:
            output_path = get_output_path(f"{identifier}_{filename}")

        try:
            print(f"  Downloading: {identifier}/{filename}")
            response = self._make_request(url, stream=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Saved to: {output_path} ({size_mb:.2f} MB)")
            return output_path

        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    def download_item(self, identifier: str, file_types: Optional[List[str]] = None,
                      max_files: int = 10) -> List[Path]:
        """
        Download files from an Internet Archive item.

        Args:
            identifier: Item identifier
            file_types: File extensions to download (e.g., ["pdf", "txt"])
            max_files: Maximum number of files to download

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"DOWNLOADING ITEM: {identifier}")
        print(f"{'='*60}")

        files = self.list_item_files(identifier)

        if not files:
            print("  No files found")
            return []

        # Filter by file type if specified
        if file_types:
            files = [f for f in files if any(f.get("name", "").lower().endswith(f".{ext}") for ext in file_types)]

        print(f"  Found {len(files)} matching files")

        downloaded = []
        for f in files[:max_files]:
            filename = f.get("name")
            if filename:
                path = self.download_ia_file(identifier, filename)
                if path:
                    downloaded.append(path)

        return downloaded

    # ==================== NEWSPAPER-SPECIFIC SEARCHES ====================

    def search_newspaper_archives(self, newspaper_domain: str,
                                  from_year: int, to_year: int,
                                  sample_per_year: int = 12) -> List[Dict]:
        """
        Search Wayback Machine for archived newspaper pages across years.

        Args:
            newspaper_domain: Newspaper domain (e.g., "nytimes.com")
            from_year: Start year
            to_year: End year
            sample_per_year: Snapshots to retrieve per year

        Returns:
            List of snapshots organized by year
        """
        print(f"\n{'='*60}")
        print(f"NEWSPAPER ARCHIVE SEARCH: {newspaper_domain}")
        print(f"Period: {from_year} - {to_year}")
        print(f"{'='*60}")

        all_results = []

        for year in range(from_year, to_year + 1):
            print(f"\n  Searching {year}...")
            results = self.search_wayback(
                url=newspaper_domain,
                from_date=f"{year}0101",
                to_date=f"{year}1231",
                limit=sample_per_year,
                match_type="domain"
            )

            for r in results:
                r["year"] = year

            all_results.extend(results)

        print(f"\n  Total snapshots found: {len(all_results)}")
        return all_results

    def download_newspaper_samples(self, newspaper_domain: str,
                                   from_year: int, to_year: int,
                                   samples_per_year: int = 1) -> List[Path]:
        """
        Download sample newspaper pages across years.

        Args:
            newspaper_domain: Newspaper domain
            from_year: Start year
            to_year: End year
            samples_per_year: Pages to download per year

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"DOWNLOADING NEWSPAPER SAMPLES: {newspaper_domain}")
        print(f"{'='*60}")

        downloaded = []

        for year in range(from_year, to_year + 1):
            results = self.search_wayback(
                url=newspaper_domain,
                from_date=f"{year}0601",  # Mid-year
                to_date=f"{year}0630",
                limit=samples_per_year,
                match_type="domain"
            )

            for snapshot in results[:samples_per_year]:
                path = self.download_wayback_page(snapshot["wayback_url"])
                if path:
                    downloaded.append(path)

        print(f"\n  Downloaded {len(downloaded)} files")
        return downloaded

    # ==================== BULK DOWNLOAD METHODS ====================

    def bulk_download_wayback(self, url: str, from_date: str, to_date: str,
                               max_downloads: int = 100,
                               file_types: Optional[List[str]] = None,
                               collapse_time: str = "timestamp:8") -> List[Path]:
        """
        Bulk download archived pages from Wayback Machine.

        Args:
            url: Website URL pattern
            from_date: Start date (YYYYMMDD)
            to_date: End date (YYYYMMDD)
            max_downloads: Maximum files to download
            file_types: Filter by mimetype (e.g., ["text/html"])
            collapse_time: Dedupe by timestamp (8=daily, 6=monthly)

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"BULK DOWNLOAD FROM WAYBACK")
        print(f"URL: {url}")
        print(f"Period: {from_date} - {to_date}")
        print(f"Max downloads: {max_downloads}")
        print(f"{'='*60}")

        # Search with collapsing to avoid duplicates
        params = {
            "url": url,
            "output": "json",
            "from": from_date,
            "to": to_date,
            "limit": max_downloads * 2,  # Get more to filter
            "collapse": collapse_time,
            "filter": "statuscode:200",
            "fl": "timestamp,original,mimetype,statuscode,digest,length"
        }

        if file_types:
            for ft in file_types:
                params["filter"] = f"mimetype:{ft}"

        try:
            response = self._make_request(self.CDX_API, params=params)
            data = response.json()

            if not data or len(data) < 2:
                print("  No results found")
                return []

            headers = data[0]
            snapshots = [dict(zip(headers, row)) for row in data[1:]]

            print(f"  Found {len(snapshots)} unique snapshots")

        except Exception as e:
            print(f"  Search failed: {e}")
            return []

        # Download files
        downloaded = []
        download_dir = get_output_path("bulk_wayback")
        download_dir.mkdir(exist_ok=True)

        for i, snapshot in enumerate(snapshots[:max_downloads]):
            timestamp = snapshot["timestamp"]
            original_url = snapshot["original"]
            wayback_url = f"{self.WAYBACK_BASE}/{timestamp}id_/{original_url}"

            # Create filename
            safe_name = original_url.replace("/", "_").replace(":", "_")[:50]
            filename = f"{timestamp}_{safe_name}.html"
            output_path = download_dir / filename

            if output_path.exists():
                print(f"  [{i+1}/{max_downloads}] Skipping (exists): {filename}")
                downloaded.append(output_path)
                continue

            print(f"  [{i+1}/{max_downloads}] Downloading: {timestamp} - {original_url[:40]}...")

            try:
                response = self._make_request(wayback_url, stream=True)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded.append(output_path)

            except Exception as e:
                print(f"    Failed: {e}")

        print(f"\n  Downloaded {len(downloaded)} files to {download_dir}")
        return downloaded

    def bulk_download_collection(self, collection: str,
                                  max_items: int = 50,
                                  file_types: Optional[List[str]] = None,
                                  max_files_per_item: int = 5) -> List[Path]:
        """
        Bulk download files from an Internet Archive collection.

        Args:
            collection: Collection identifier
            max_items: Maximum items to download from
            file_types: File extensions to download (e.g., ["pdf", "txt", "jp2"])
            max_files_per_item: Max files per item

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"BULK DOWNLOAD FROM COLLECTION: {collection}")
        print(f"Max items: {max_items}")
        print(f"{'='*60}")

        # Get items from collection
        items = self.scrape_collection(collection, count=max_items)

        if not items:
            print("  No items found")
            return []

        downloaded = []
        download_dir = get_output_path(f"collection_{collection}")
        download_dir.mkdir(exist_ok=True)

        for i, item in enumerate(items):
            identifier = item.get("identifier")
            if not identifier:
                continue

            print(f"\n  [{i+1}/{len(items)}] Processing: {identifier}")

            # Get files for this item
            files = self.list_item_files(identifier)

            if file_types:
                files = [f for f in files if any(
                    f.get("name", "").lower().endswith(f".{ext}")
                    for ext in file_types
                )]

            # Download files
            for j, file_info in enumerate(files[:max_files_per_item]):
                filename = file_info.get("name")
                if not filename:
                    continue

                output_path = download_dir / f"{identifier}_{filename}"

                if output_path.exists():
                    print(f"    Skipping (exists): {filename}")
                    downloaded.append(output_path)
                    continue

                url = f"{self.IA_DOWNLOAD_BASE}/{identifier}/{filename}"

                try:
                    print(f"    Downloading: {filename}")
                    response = self._make_request(url, stream=True)

                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded.append(output_path)

                except Exception as e:
                    print(f"    Failed: {e}")

        print(f"\n  Downloaded {len(downloaded)} files to {download_dir}")
        return downloaded

    def bulk_download_newspaper_timeline(self, newspaper_domain: str,
                                          from_year: int, to_year: int,
                                          samples_per_month: int = 1) -> List[Path]:
        """
        Download newspaper samples across a timeline.

        Args:
            newspaper_domain: Newspaper domain (e.g., "nytimes.com")
            from_year: Start year
            to_year: End year
            samples_per_month: Snapshots per month

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"BULK DOWNLOAD NEWSPAPER TIMELINE")
        print(f"Newspaper: {newspaper_domain}")
        print(f"Period: {from_year} - {to_year}")
        print(f"{'='*60}")

        downloaded = []
        download_dir = get_output_path(f"newspaper_{newspaper_domain.replace('.', '_')}")
        download_dir.mkdir(exist_ok=True)

        total_months = (to_year - from_year + 1) * 12
        month_count = 0

        for year in range(from_year, to_year + 1):
            for month in range(1, 13):
                month_count += 1
                from_date = f"{year}{month:02d}01"
                to_date = f"{year}{month:02d}28"

                print(f"\n  [{month_count}/{total_months}] {year}-{month:02d}")

                # Search for snapshots this month
                results = self.search_wayback(
                    url=newspaper_domain,
                    from_date=from_date,
                    to_date=to_date,
                    limit=samples_per_month,
                    match_type="host"
                )

                for snapshot in results[:samples_per_month]:
                    timestamp = snapshot["timestamp"]
                    wayback_url = snapshot["wayback_url"]

                    filename = f"{year}_{month:02d}_{timestamp}.html"
                    output_path = download_dir / filename

                    if output_path.exists():
                        downloaded.append(output_path)
                        continue

                    try:
                        response = self._make_request(wayback_url, stream=True)
                        with open(output_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        downloaded.append(output_path)
                        print(f"    Downloaded: {filename}")

                    except Exception as e:
                        print(f"    Failed: {e}")

        print(f"\n  Total downloaded: {len(downloaded)} files to {download_dir}")
        return downloaded

    def download_full_item(self, identifier: str,
                           file_types: Optional[List[str]] = None) -> List[Path]:
        """
        Download ALL files from an Internet Archive item.

        Args:
            identifier: Item identifier
            file_types: Optional filter by extension

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"FULL ITEM DOWNLOAD: {identifier}")
        print(f"{'='*60}")

        # Get item metadata
        metadata = self.get_item_metadata(identifier)
        if not metadata:
            print("  Item not found")
            return []

        files = metadata.get("files", [])
        item_title = metadata.get("metadata", {}).get("title", identifier)

        print(f"  Title: {item_title}")
        print(f"  Total files: {len(files)}")

        if file_types:
            files = [f for f in files if any(
                f.get("name", "").lower().endswith(f".{ext}")
                for ext in file_types
            )]
            print(f"  Filtered files: {len(files)}")

        # Calculate total size
        total_size = sum(int(f.get("size", 0)) for f in files)
        print(f"  Total size: {total_size / (1024*1024):.2f} MB")

        downloaded = []
        download_dir = get_output_path(f"item_{identifier}")
        download_dir.mkdir(exist_ok=True)

        for i, file_info in enumerate(files):
            filename = file_info.get("name")
            if not filename:
                continue

            output_path = download_dir / filename
            file_size = int(file_info.get("size", 0))

            if output_path.exists() and output_path.stat().st_size == file_size:
                print(f"  [{i+1}/{len(files)}] Skipping (exists): {filename}")
                downloaded.append(output_path)
                continue

            url = f"{self.IA_DOWNLOAD_BASE}/{identifier}/{quote(filename)}"

            try:
                print(f"  [{i+1}/{len(files)}] Downloading: {filename} ({file_size/1024:.1f} KB)")
                response = self._make_request(url, stream=True)

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded.append(output_path)

            except Exception as e:
                print(f"    Failed: {e}")

        print(f"\n  Downloaded {len(downloaded)}/{len(files)} files to {download_dir}")
        return downloaded

    # ==================== TEXT EXTRACTION ====================

    def extract_text_from_html(self, html_path: Path,
                                remove_scripts: bool = True,
                                remove_styles: bool = True) -> Optional[str]:
        """
        Extract readable text from an HTML file.

        Args:
            html_path: Path to HTML file
            remove_scripts: Remove script tags
            remove_styles: Remove style tags

        Returns:
            Extracted text or None
        """
        if not HAS_BS4:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return None

        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "lxml")

            # Remove unwanted elements
            if remove_scripts:
                for script in soup(["script", "noscript"]):
                    script.decompose()

            if remove_styles:
                for style in soup(["style"]):
                    style.decompose()

            # Remove hidden elements
            for hidden in soup.find_all(style=re.compile(r"display:\s*none")):
                hidden.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            return text

        except Exception as e:
            print(f"  Error extracting text: {e}")
            return None

    def extract_article_content(self, html_path: Path, include_full_text: bool = True) -> Dict:
        """
        Extract structured content from a newspaper HTML page.

        Args:
            html_path: Path to HTML file
            include_full_text: Whether to include the full article text

        Returns:
            Dictionary with title, headlines, full_text, articles, etc.
        """
        if not HAS_BS4:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return {}

        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "lxml")

            # Extract timestamp from filename
            timestamp = html_path.stem.split("_")[0]
            date_str = ""
            if len(timestamp) >= 8 and timestamp[:8].isdigit():
                date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

            content = {
                "file": str(html_path.name),
                "date": date_str,
                "title": "",
                "headlines": [],
                "full_text": "",
                "paragraphs": [],
                "links": [],
                "meta_description": "",
                "word_count": 0
            }

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                content["title"] = title_tag.get_text(strip=True)

            # Extract meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                content["meta_description"] = meta_desc.get("content", "")

            # Extract headlines (h1, h2, h3)
            for tag in ["h1", "h2", "h3"]:
                for heading in soup.find_all(tag):
                    text = heading.get_text(strip=True)
                    if text and len(text) > 10:
                        content["headlines"].append(text)

            # Extract ALL paragraphs (full text, not snippets)
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 50:  # Filter out very short paragraphs
                    content["paragraphs"].append(text)

            # Extract news links
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                text = a.get_text(strip=True)
                # Match year patterns in URLs
                if any(f"/{year}/" in href for year in range(2020, 2030)):
                    if text and len(text) > 20:
                        content["links"].append({"text": text, "href": href})

            # Get full cleaned text
            if include_full_text:
                full_text = self.extract_text_from_html(html_path)
                if full_text:
                    content["full_text"] = full_text
                    content["word_count"] = len(full_text.split())

            return content

        except Exception as e:
            print(f"  Error extracting content: {e}")
            return {}

    def bulk_extract_text(self, input_dir: Path,
                          output_dir: Optional[Path] = None,
                          file_pattern: str = "*.html") -> List[Path]:
        """
        Extract text from all HTML files in a directory.

        Args:
            input_dir: Directory containing HTML files
            output_dir: Where to save text files (default: same dir with _text suffix)
            file_pattern: Glob pattern for files

        Returns:
            List of created text file paths
        """
        print(f"\n{'='*60}")
        print(f"BULK TEXT EXTRACTION")
        print(f"Input: {input_dir}")
        print(f"{'='*60}")

        if not HAS_BS4:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return []

        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_text"

        output_dir.mkdir(exist_ok=True)

        html_files = list(input_dir.glob(file_pattern))
        print(f"  Found {len(html_files)} HTML files")

        extracted = []
        for i, html_path in enumerate(html_files):
            print(f"  [{i+1}/{len(html_files)}] Processing: {html_path.name}")

            text = self.extract_text_from_html(html_path)
            if text:
                output_path = output_dir / f"{html_path.stem}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                extracted.append(output_path)
                print(f"    Extracted {len(text.split())} words")

        print(f"\n  Extracted text from {len(extracted)} files to {output_dir}")
        return extracted

    def bulk_extract_articles(self, input_dir: Path,
                               output_path: Optional[Path] = None,
                               include_full_text: bool = True) -> Dict:
        """
        Extract structured article data from all HTML files with FULL TEXT.

        Args:
            input_dir: Directory containing HTML files
            output_path: Where to save JSON results
            include_full_text: Include the complete article text in output

        Returns:
            Dictionary with all extracted data including full article text
        """
        print(f"\n{'='*60}")
        print(f"BULK ARTICLE EXTRACTION (Full Text)")
        print(f"Input: {input_dir}")
        print(f"{'='*60}")

        if not HAS_BS4:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return {}

        html_files = sorted(input_dir.glob("*.html"))
        print(f"  Found {len(html_files)} HTML files")

        all_data = {
            "extraction_date": datetime.now().isoformat(),
            "source_dir": str(input_dir),
            "file_count": len(html_files),
            "articles": [],
            "all_headlines": [],
            "total_words": 0,
            "total_paragraphs": 0
        }

        for i, html_path in enumerate(html_files):
            print(f"  [{i+1}/{len(html_files)}] Processing: {html_path.name}")

            content = self.extract_article_content(html_path, include_full_text=include_full_text)
            if content:
                all_data["articles"].append(content)
                all_data["all_headlines"].extend(content.get("headlines", []))
                all_data["total_words"] += content.get("word_count", 0)
                all_data["total_paragraphs"] += len(content.get("paragraphs", []))

        # Summary stats
        all_data["unique_headlines"] = len(set(all_data["all_headlines"]))
        all_data["avg_words_per_page"] = (
            all_data["total_words"] // len(html_files) if html_files else 0
        )

        print(f"\n  Summary:")
        print(f"    Total articles: {len(all_data['articles'])}")
        print(f"    Total headlines: {len(all_data['all_headlines'])}")
        print(f"    Unique headlines: {all_data['unique_headlines']}")
        print(f"    Total paragraphs: {all_data['total_paragraphs']:,}")
        print(f"    Total words: {all_data['total_words']:,}")
        print(f"    Avg words/page: {all_data['avg_words_per_page']:,}")

        # Save results
        if output_path is None:
            output_path = get_output_path(f"full_articles_{input_dir.name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Results saved to: {output_path}")
        print(f"  File size: {file_size:.2f} MB")
        return all_data

    def create_text_corpus(self, input_dir: Path,
                           output_path: Optional[Path] = None) -> Path:
        """
        Create a single text corpus file from all HTML files.

        Args:
            input_dir: Directory containing HTML files
            output_path: Output file path

        Returns:
            Path to corpus file
        """
        print(f"\n{'='*60}")
        print(f"CREATING TEXT CORPUS")
        print(f"Input: {input_dir}")
        print(f"{'='*60}")

        if not HAS_BS4:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return None

        html_files = sorted(input_dir.glob("*.html"))
        print(f"  Found {len(html_files)} HTML files")

        if output_path is None:
            output_path = get_output_path(f"corpus_{input_dir.name}.txt")

        total_words = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, html_path in enumerate(html_files):
                # Extract date from filename (format: YYYYMMDD...)
                timestamp = html_path.stem.split("_")[0]
                date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

                out_f.write(f"\n{'='*60}\n")
                out_f.write(f"DATE: {date_str}\n")
                out_f.write(f"FILE: {html_path.name}\n")
                out_f.write(f"{'='*60}\n\n")

                text = self.extract_text_from_html(html_path)
                if text:
                    out_f.write(text)
                    out_f.write("\n\n")
                    total_words += len(text.split())

                print(f"  [{i+1}/{len(html_files)}] Added: {date_str}")

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Corpus created:")
        print(f"    Path: {output_path}")
        print(f"    Size: {file_size:.2f} MB")
        print(f"    Total words: {total_words:,}")

        return output_path

    def download_articles_from_links(self, json_path: Path,
                                       max_articles: int = 50,
                                       output_dir: Optional[Path] = None) -> List[Path]:
        """
        Download individual article pages from links extracted from homepage.

        Args:
            json_path: Path to JSON file with extracted article links
            max_articles: Maximum articles to download
            output_dir: Where to save downloaded articles

        Returns:
            List of downloaded file paths
        """
        print(f"\n{'='*60}")
        print(f"DOWNLOADING INDIVIDUAL ARTICLES")
        print(f"{'='*60}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect all unique article URLs
        all_links = []
        seen_urls = set()
        for article in data.get("articles", []):
            for link in article.get("links", []):
                href = link.get("href", "")
                # Filter for actual article URLs (contain year pattern, not interactive)
                if href and href not in seen_urls:
                    if any(f"/{year}/" in href for year in range(2020, 2030)):
                        if "interactive" not in href and ".html" in href:
                            seen_urls.add(href)
                            all_links.append(link)

        print(f"  Found {len(all_links)} unique article URLs")

        if output_dir is None:
            output_dir = get_output_path("articles_full")
        output_dir.mkdir(exist_ok=True)

        downloaded = []
        for i, link in enumerate(all_links[:max_articles]):
            original_url = link["href"]
            title = link["text"][:50]

            # Create safe filename
            url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
            # Extract date from URL if possible
            date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", original_url)
            if date_match:
                date_str = f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
            else:
                date_str = "nodate"

            filename = f"{date_str}_{url_hash}.html"
            output_path = output_dir / filename

            if output_path.exists():
                print(f"  [{i+1}/{min(len(all_links), max_articles)}] Skipping (exists): {title}...")
                downloaded.append(output_path)
                continue

            # Try to get from Wayback first
            print(f"  [{i+1}/{min(len(all_links), max_articles)}] Downloading: {title}...")

            # Check Wayback availability
            availability = self.check_availability(original_url)
            if availability:
                wayback_url = availability.get("url")
                try:
                    response = self._make_request(wayback_url, stream=True)
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded.append(output_path)
                    print(f"    Saved from Wayback")
                except Exception as e:
                    print(f"    Failed: {e}")
            else:
                print(f"    Not available in Wayback")

        print(f"\n  Downloaded {len(downloaded)} articles to {output_dir}")
        return downloaded

    def _extract_text_with_spaces(self, element) -> str:
        """
        Extract text from HTML element, preserving spaces between inline elements.

        This fixes the issue where BeautifulSoup's get_text() concatenates
        text from adjacent elements without spaces (e.g., hyperlinks).
        """
        if element is None:
            return ""

        # Get all text nodes and inline elements
        texts = []
        for item in element.descendants:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    texts.append(text)
            elif item.name in ['br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4']:
                # Block elements get newlines
                if texts and texts[-1] != '\n':
                    texts.append('\n')

        # Join with spaces, then clean up
        result = ' '.join(texts)
        # Fix multiple spaces
        result = re.sub(r' +', ' ', result)
        # Fix space before punctuation
        result = re.sub(r' ([.,;:!?])', r'\1', result)
        # Fix newlines
        result = re.sub(r'\n +', '\n', result)
        result = re.sub(r' +\n', '\n', result)

        return result.strip()

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by fixing common issues.
        """
        # Fix missing spaces before capital letters (camelCase from links)
        # e.g., "wordWord" -> "word Word"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Fix missing space after punctuation (but not in URLs or numbers)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

        # Fix quotes without spaces
        text = re.sub(r'([a-z])"([A-Z])', r'\1" \2', text)
        text = re.sub(r'([.,])"([a-z])', r'\1" \2', text)

        # Fix "By" prefix in author lines
        text = re.sub(r'^By([A-Z])', r'By \1', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def _extract_author(self, soup) -> str:
        """
        Extract clean author name from HTML.
        """
        # Try meta tag first (usually cleanest)
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag:
            author = author_tag.get("content", "")
            if author:
                # Clean up "By " prefix and duplicates
                author = re.sub(r'^By\s*', '', author)
                return author

        # Try specific NYTimes byline patterns
        byline = soup.find("span", class_=re.compile(r"byline-prefix", re.I))
        if byline:
            # Get just the name, not the bio
            name_elem = byline.find_next("a") or byline.find_next("span")
            if name_elem:
                return name_elem.get_text(strip=True)

        # Fallback to general byline
        byline = soup.find(class_=re.compile(r"^byline$", re.I))
        if byline:
            # Try to get just names (usually in links)
            names = []
            for a in byline.find_all("a"):
                name = a.get_text(strip=True)
                if name and len(name) < 50:  # Reasonable name length
                    names.append(name)
            if names:
                return ", ".join(names)

        return ""

    def extract_full_articles_json(self, input_dir: Path,
                                    output_path: Optional[Path] = None) -> Dict:
        """
        Extract full article text from downloaded article pages and save as JSON.

        Each article will have:
        - file: source filename
        - date: extracted date
        - title: article title
        - author: article author (if found)
        - full_text: complete article text (cleaned)
        - word_count: number of words

        Args:
            input_dir: Directory containing HTML article files
            output_path: Where to save JSON results

        Returns:
            Dictionary with all extracted articles
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING FULL ARTICLE TEXT TO JSON")
        print(f"Input: {input_dir}")
        print(f"{'='*60}")

        if not HAS_BS4:
            print("BeautifulSoup not installed.")
            return {}

        html_files = sorted(input_dir.glob("*.html"))
        print(f"  Found {len(html_files)} HTML files")

        all_data = {
            "extraction_date": datetime.now().isoformat(),
            "source_dir": str(input_dir),
            "article_count": 0,
            "total_words": 0,
            "articles": []
        }

        for i, html_path in enumerate(html_files):
            print(f"  [{i+1}/{len(html_files)}] Processing: {html_path.name}")

            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, "lxml")

                # Extract date from filename
                date_match = re.match(r"(\d{8})", html_path.stem)
                date_str = ""
                if date_match:
                    d = date_match.group(1)
                    date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

                article = {
                    "file": html_path.name,
                    "date": date_str,
                    "title": "",
                    "author": "",
                    "section": "",
                    "full_text": "",
                    "word_count": 0
                }

                # Extract title
                title_tag = soup.find("title")
                if title_tag:
                    article["title"] = title_tag.get_text(strip=True)

                # Try to find article headline (h1)
                h1 = soup.find("h1")
                if h1:
                    headline = h1.get_text(strip=True)
                    if len(headline) > len(article["title"]):
                        article["title"] = headline

                # Extract author (using improved method)
                article["author"] = self._extract_author(soup)

                # Extract section
                section_tag = soup.find("meta", attrs={"property": "article:section"})
                if section_tag:
                    article["section"] = section_tag.get("content", "")

                # Extract full article text
                # Remove unwanted elements first
                for tag in soup(["script", "noscript", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()

                # Find article body (common container classes)
                article_body = soup.find("article") or soup.find(class_=re.compile(r"article|story|content", re.I))

                if article_body:
                    # Get all paragraphs from article body with proper spacing
                    paragraphs = []
                    for p in article_body.find_all("p"):
                        # Use improved text extraction
                        text = self._extract_text_with_spaces(p)
                        text = self._clean_text(text)
                        if len(text) > 30:  # Filter short paragraphs
                            paragraphs.append(text)
                    article["full_text"] = "\n\n".join(paragraphs)
                else:
                    # Fallback: get all paragraphs
                    paragraphs = []
                    for p in soup.find_all("p"):
                        text = self._extract_text_with_spaces(p)
                        text = self._clean_text(text)
                        if len(text) > 50:
                            paragraphs.append(text)
                    article["full_text"] = "\n\n".join(paragraphs)

                article["word_count"] = len(article["full_text"].split())

                if article["word_count"] > 100:  # Only include substantial articles
                    all_data["articles"].append(article)
                    all_data["total_words"] += article["word_count"]
                    print(f"    Title: {article['title'][:60]}...")
                    print(f"    Words: {article['word_count']}")

            except Exception as e:
                print(f"    Error: {e}")

        all_data["article_count"] = len(all_data["articles"])
        all_data["avg_words"] = all_data["total_words"] // max(1, all_data["article_count"])

        print(f"\n  Summary:")
        print(f"    Articles extracted: {all_data['article_count']}")
        print(f"    Total words: {all_data['total_words']:,}")
        print(f"    Avg words/article: {all_data['avg_words']:,}")

        if output_path is None:
            output_path = get_output_path(f"newspaper_articles_{input_dir.name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Saved to: {output_path}")
        print(f"  File size: {file_size:.2f} MB")

        return all_data

    # ==================== REPORTS AND SUMMARIES ====================

    def generate_availability_report(self, urls: List[str],
                                     output_path: Optional[Path] = None) -> Dict:
        """
        Generate a report on Wayback availability for multiple URLs.

        Args:
            urls: List of URLs to check
            output_path: Where to save the report

        Returns:
            Report dictionary
        """
        print(f"\n{'='*60}")
        print("AVAILABILITY REPORT")
        print(f"{'='*60}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "urls_checked": len(urls),
            "results": []
        }

        for url in urls:
            print(f"\n  Checking: {url}")
            availability = self.check_availability(url)

            result = {
                "url": url,
                "available": availability is not None,
            }

            if availability:
                result["snapshot_url"] = availability.get("url")
                result["snapshot_timestamp"] = availability.get("timestamp")
                result["status"] = availability.get("status")
                print(f"    Available: {availability.get('timestamp')}")
            else:
                print("    Not available")

            report["results"].append(result)

        # Summary
        available_count = sum(1 for r in report["results"] if r["available"])
        report["summary"] = {
            "total": len(urls),
            "available": available_count,
            "unavailable": len(urls) - available_count
        }

        print(f"\n  Summary: {available_count}/{len(urls)} URLs available")

        if output_path is None:
            output_path = get_output_path("availability_report.json")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {output_path}")

        return report


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Demonstrate the newspaper downloader."""
    print_header("WAYBACK MACHINE NEWSPAPER DOWNLOADER")

    downloader = WaybackNewspaperDownloader()

    # Example 1: Search Wayback for a newspaper
    print_header("EXAMPLE 1: Search Wayback for NYTimes archives")
    results = downloader.search_wayback(
        url="nytimes.com",
        from_date="20100101",
        to_date="20101231",
        limit=10
    )

    if results:
        print("\nSample snapshots:")
        for r in results[:5]:
            print(f"  {r['timestamp']}: {r['original'][:60]}...")

    # Example 2: Check availability
    print_header("EXAMPLE 2: Check URL availability")
    availability = downloader.check_availability("nytimes.com/2020/01/01")
    if availability:
        print(f"  Available: {availability}")

    # Example 3: Search Internet Archive newspapers
    print_header("EXAMPLE 3: Search IA newspaper collections")
    ia_results = downloader.search_ia_newspapers(
        query="newspaper",
        rows=10
    )

    # Example 4: Generate availability report
    print_header("EXAMPLE 4: Availability report for major newspapers")
    newspaper_urls = [
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "latimes.com",
        "chicagotribune.com"
    ]
    downloader.generate_availability_report(newspaper_urls)

    print_header("EXAMPLES COMPLETE")
    print("""
Available methods:

WAYBACK MACHINE:
  - search_wayback(url, from_date, to_date)  : Search for archived snapshots
  - check_availability(url)                   : Check if URL is archived
  - download_wayback_page(wayback_url)        : Download archived page

INTERNET ARCHIVE:
  - search_ia_newspapers(query)               : Search IA newspaper collections
  - scrape_collection(collection)             : Get metadata from collection
  - get_item_metadata(identifier)             : Get full item metadata
  - download_ia_file(identifier, filename)    : Download specific file
  - download_item(identifier)                 : Download all files from item

BULK DOWNLOAD:
  - bulk_download_wayback(url, dates)         : Bulk download from Wayback
  - bulk_download_collection(collection)      : Download from IA collection
  - bulk_download_newspaper_timeline(domain)  : Download across years
  - download_full_item(identifier)            : Download entire IA item

NEWSPAPER HELPERS:
  - search_newspaper_archives(domain, years)  : Search across years
  - download_newspaper_samples(domain, years) : Download samples
  - generate_availability_report(urls)        : Check multiple URLs

BULK DOWNLOAD EXAMPLES:
  # Download 50 NYTimes snapshots from 2010
  downloader.bulk_download_wayback("nytimes.com", "20100101", "20101231", max_downloads=50)

  # Download from newspaper collection
  downloader.bulk_download_collection("newspapers", max_items=10, file_types=["pdf"])

  # Download monthly samples across years
  downloader.bulk_download_newspaper_timeline("washingtonpost.com", 2015, 2020)

  # Download entire archive item
  downloader.download_full_item("some_newspaper_identifier", file_types=["pdf", "txt"])
""")


if __name__ == "__main__":
    main()
