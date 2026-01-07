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
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import quote, urljoin
import hashlib

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
