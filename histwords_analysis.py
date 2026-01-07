"""
HistWords Analysis Toolkit
==========================
A comprehensive toolkit for analyzing historical word embeddings from HistWords.

Features:
- Semantic change tracking over time
- Word similarity analysis across time periods
- Word analogy computations

Usage:
    from histwords_analysis import HistWordsAnalyzer

    analyzer = HistWordsAnalyzer("eng-all_sgns")
    analyzer.semantic_change("computer")
    analyzer.similar_words("woman", year=1900)
    analyzer.analogy("king", "man", "woman", year=1950)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

# Optional imports for visualization and clustering
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Output directory for saving plots
OUTPUT_DIR = Path(__file__).parent / "output"


def get_output_path(filename: str) -> Path:
    """Get path in output directory, creating it if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


class HistWordsAnalyzer:
    """Main class for analyzing HistWords embeddings."""

    def __init__(self, dataset_path: str, base_dir: Optional[str] = None):
        """
        Initialize the analyzer with a dataset.

        Args:
            dataset_path: Name of the dataset folder (e.g., "eng-all_sgns")
            base_dir: Base directory containing datasets. Defaults to script directory.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent

        self.base_dir = Path(base_dir)
        self.dataset_path = self.base_dir / dataset_path
        self.dataset_name = dataset_path

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {self.dataset_path}")

        # Cache for loaded embeddings and vocabularies
        self._embeddings: Dict[int, np.ndarray] = {}
        self._vocab: Dict[int, Dict[str, int]] = {}
        self._index_to_word: Dict[int, Dict[int, str]] = {}

        # Discover available years
        self.years = self._discover_years()
        print(f"Loaded dataset: {dataset_path}")
        print(f"Available years: {self.years}")

    def _discover_years(self) -> List[int]:
        """Discover all available years in the dataset."""
        years = set()
        for f in self.dataset_path.glob("*-vocab.pkl"):
            year = int(f.stem.split("-")[0])
            years.add(year)
        return sorted(years)

    def _load_year(self, year: int) -> None:
        """Load embeddings and vocabulary for a specific year."""
        if year in self._embeddings:
            return

        if year not in self.years:
            raise ValueError(f"Year {year} not available. Available years: {self.years}")

        vocab_path = self.dataset_path / f"{year}-vocab.pkl"
        embed_path = self.dataset_path / f"{year}-w.npy"

        # Load vocabulary (stored as a list where index = embedding row)
        with open(vocab_path, "rb") as f:
            vocab_list = pickle.load(f)

        # Convert list to word -> index dict
        self._vocab[year] = {word: idx for idx, word in enumerate(vocab_list)}

        # Create reverse mapping (index -> word)
        self._index_to_word[year] = {idx: word for idx, word in enumerate(vocab_list)}

        # Load embeddings
        self._embeddings[year] = np.load(embed_path)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self._embeddings[year], axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self._embeddings[year] = self._embeddings[year] / norms

    def get_vector(self, word: str, year: int) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word in a specific year.

        Args:
            word: The word to look up
            year: The year/decade

        Returns:
            Normalized embedding vector or None if word not found
        """
        self._load_year(year)

        if word not in self._vocab[year]:
            return None

        idx = self._vocab[year][word]
        return self._embeddings[year][idx]

    def word_exists(self, word: str, year: Optional[int] = None) -> Union[bool, Dict[int, bool]]:
        """
        Check if a word exists in the vocabulary.

        Args:
            word: The word to check
            year: Specific year to check, or None to check all years

        Returns:
            Boolean if year specified, otherwise dict of year -> bool
        """
        if year is not None:
            self._load_year(year)
            return word in self._vocab[year]

        result = {}
        for y in self.years:
            self._load_year(y)
            result[y] = word in self._vocab[y]
        return result

    # ==================== SEMANTIC CHANGE ANALYSIS ====================

    def semantic_change(self, word: str, method: str = "cosine") -> Dict[str, any]:
        """
        Track semantic change of a word over time.

        Args:
            word: The word to analyze
            method: "cosine" for cosine distance, "neighbors" for neighborhood change

        Returns:
            Dictionary with change metrics and details
        """
        print(f"\n{'='*60}")
        print(f"SEMANTIC CHANGE ANALYSIS: '{word}'")
        print(f"{'='*60}")

        # Find years where word exists
        available_years = []
        for year in self.years:
            self._load_year(year)
            if word in self._vocab[year]:
                available_years.append(year)

        if len(available_years) < 2:
            print(f"Word '{word}' found in fewer than 2 time periods.")
            return {"error": "Insufficient data", "available_years": available_years}

        print(f"Word found in years: {available_years}")

        if method == "cosine":
            return self._semantic_change_cosine(word, available_years)
        elif method == "neighbors":
            return self._semantic_change_neighbors(word, available_years)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _semantic_change_cosine(self, word: str, years: List[int]) -> Dict:
        """Measure semantic change using cosine similarity between consecutive periods."""
        vectors = {y: self.get_vector(word, y) for y in years}

        # Calculate pairwise similarities
        similarities = []
        print(f"\nCosine similarity between consecutive periods:")
        print("-" * 40)

        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            sim = float(np.dot(vectors[y1], vectors[y2]))
            similarities.append({
                "from_year": y1,
                "to_year": y2,
                "similarity": sim,
                "change": 1 - sim
            })
            print(f"  {y1} -> {y2}: similarity={sim:.4f}, change={1-sim:.4f}")

        # Overall change (first to last)
        first_year, last_year = years[0], years[-1]
        overall_sim = float(np.dot(vectors[first_year], vectors[last_year]))

        print(f"\nOverall change ({first_year} -> {last_year}):")
        print(f"  Similarity: {overall_sim:.4f}")
        print(f"  Total change: {1 - overall_sim:.4f}")

        # Find period of greatest change
        max_change = max(similarities, key=lambda x: x["change"])
        print(f"\nGreatest change period: {max_change['from_year']} -> {max_change['to_year']}")
        print(f"  Change magnitude: {max_change['change']:.4f}")

        return {
            "word": word,
            "method": "cosine",
            "years_analyzed": years,
            "pairwise_changes": similarities,
            "overall_similarity": overall_sim,
            "overall_change": 1 - overall_sim,
            "max_change_period": max_change
        }

    def _semantic_change_neighbors(self, word: str, years: List[int], k: int = 10) -> Dict:
        """Measure semantic change using neighborhood overlap."""
        print(f"\nNeighborhood change analysis (top {k} neighbors):")
        print("-" * 40)

        neighbor_sets = {}
        for year in years:
            neighbors = self.similar_words(word, year, top_k=k, verbose=False)
            neighbor_sets[year] = set([n[0] for n in neighbors])

        # Calculate Jaccard similarity between consecutive periods
        overlaps = []
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            intersection = neighbor_sets[y1] & neighbor_sets[y2]
            union = neighbor_sets[y1] | neighbor_sets[y2]
            jaccard = len(intersection) / len(union) if union else 0

            overlaps.append({
                "from_year": y1,
                "to_year": y2,
                "jaccard_similarity": jaccard,
                "common_neighbors": list(intersection),
                "lost_neighbors": list(neighbor_sets[y1] - neighbor_sets[y2]),
                "gained_neighbors": list(neighbor_sets[y2] - neighbor_sets[y1])
            })

            print(f"\n{y1} -> {y2}:")
            print(f"  Jaccard similarity: {jaccard:.4f}")
            print(f"  Common: {list(intersection)[:5]}...")
            print(f"  Lost: {list(neighbor_sets[y1] - neighbor_sets[y2])[:5]}...")
            print(f"  Gained: {list(neighbor_sets[y2] - neighbor_sets[y1])[:5]}...")

        return {
            "word": word,
            "method": "neighbors",
            "k": k,
            "years_analyzed": years,
            "neighbor_changes": overlaps,
            "neighbor_sets": {y: list(s) for y, s in neighbor_sets.items()}
        }

    def compare_semantic_change(self, words: List[str]) -> Dict:
        """
        Compare semantic change across multiple words.

        Args:
            words: List of words to compare

        Returns:
            Comparison results ranked by change magnitude
        """
        print(f"\n{'='*60}")
        print(f"COMPARING SEMANTIC CHANGE: {words}")
        print(f"{'='*60}")

        results = []
        for word in words:
            change_data = self.semantic_change(word, method="cosine")
            if "error" not in change_data:
                results.append({
                    "word": word,
                    "overall_change": change_data["overall_change"],
                    "years_covered": f"{change_data['years_analyzed'][0]}-{change_data['years_analyzed'][-1]}"
                })

        # Sort by change magnitude
        results.sort(key=lambda x: x["overall_change"], reverse=True)

        print(f"\n{'='*60}")
        print("RANKING BY SEMANTIC CHANGE (most to least):")
        print("-" * 40)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['word']}: {r['overall_change']:.4f} ({r['years_covered']})")

        return {"ranking": results}

    # ==================== WORD SIMILARITY ANALYSIS ====================

    def similar_words(self, word: str, year: int, top_k: int = 10,
                      verbose: bool = True) -> List[Tuple[str, float]]:
        """
        Find most similar words in a given year.

        Args:
            word: The query word
            year: The year to search in
            top_k: Number of similar words to return
            verbose: Whether to print results

        Returns:
            List of (word, similarity) tuples
        """
        self._load_year(year)

        if word not in self._vocab[year]:
            if verbose:
                print(f"Word '{word}' not found in year {year}")
            return []

        vec = self.get_vector(word, year)

        # Compute similarities with all words
        similarities = np.dot(self._embeddings[year], vec)

        # Get top-k (excluding the word itself)
        top_indices = np.argsort(similarities)[::-1][:top_k + 1]

        results = []
        for idx in top_indices:
            similar_word = self._index_to_word[year][idx]
            if similar_word != word:
                results.append((similar_word, float(similarities[idx])))
            if len(results) >= top_k:
                break

        if verbose:
            print(f"\n{'='*60}")
            print(f"SIMILAR WORDS TO '{word}' in {year}")
            print(f"{'='*60}")
            for i, (w, sim) in enumerate(results, 1):
                print(f"  {i:2}. {w:20} {sim:.4f}")

        return results

    def similarity_over_time(self, word1: str, word2: str) -> Dict[int, float]:
        """
        Track similarity between two words over time.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Dictionary mapping year to similarity score
        """
        print(f"\n{'='*60}")
        print(f"SIMILARITY OVER TIME: '{word1}' vs '{word2}'")
        print(f"{'='*60}")

        results = {}
        for year in self.years:
            vec1 = self.get_vector(word1, year)
            vec2 = self.get_vector(word2, year)

            if vec1 is not None and vec2 is not None:
                sim = float(np.dot(vec1, vec2))
                results[year] = sim
                print(f"  {year}: {sim:.4f}")
            else:
                missing = []
                if vec1 is None:
                    missing.append(word1)
                if vec2 is None:
                    missing.append(word2)
                print(f"  {year}: N/A (missing: {missing})")

        if results:
            max_year = max(results, key=results.get)
            min_year = min(results, key=results.get)
            print(f"\nMost similar in {max_year}: {results[max_year]:.4f}")
            print(f"Least similar in {min_year}: {results[min_year]:.4f}")

        return results

    def neighborhood_comparison(self, word: str, years: List[int], top_k: int = 10) -> Dict:
        """
        Compare word neighborhoods across different years.

        Args:
            word: The word to analyze
            years: List of years to compare
            top_k: Number of neighbors to consider

        Returns:
            Comparison data including common and unique neighbors
        """
        print(f"\n{'='*60}")
        print(f"NEIGHBORHOOD COMPARISON: '{word}'")
        print(f"Years: {years}")
        print(f"{'='*60}")

        neighborhoods = {}
        for year in years:
            neighbors = self.similar_words(word, year, top_k=top_k, verbose=False)
            if neighbors:
                neighborhoods[year] = {w: s for w, s in neighbors}

        # Find words present in all years
        if len(neighborhoods) > 1:
            common = set.intersection(*[set(n.keys()) for n in neighborhoods.values()])
            print(f"\nWords in top-{top_k} across ALL periods ({len(common)}):")
            for w in list(common)[:10]:
                scores = [f"{y}:{neighborhoods[y][w]:.3f}" for y in sorted(neighborhoods.keys())]
                print(f"  {w}: {', '.join(scores)}")

        # Show unique neighbors per period
        print(f"\nUnique top neighbors per period:")
        for year in sorted(neighborhoods.keys()):
            other_words = set()
            for y, n in neighborhoods.items():
                if y != year:
                    other_words.update(n.keys())
            unique = set(neighborhoods[year].keys()) - other_words
            print(f"  {year}: {list(unique)[:5]}")

        return {
            "word": word,
            "years": years,
            "neighborhoods": neighborhoods,
            "common_neighbors": list(common) if len(neighborhoods) > 1 else []
        }

    # ==================== WORD ANALOGY ANALYSIS ====================

    def analogy(self, a: str, b: str, c: str, year: int, top_k: int = 5,
                verbose: bool = True) -> List[Tuple[str, float]]:
        """
        Compute word analogies: a is to b as c is to ?

        Uses the classic formula: vec(?) = vec(b) - vec(a) + vec(c)

        Args:
            a, b, c: Words for the analogy
            year: The year to use
            top_k: Number of results to return
            verbose: Whether to print results

        Returns:
            List of (word, similarity) tuples
        """
        self._load_year(year)

        # Check all words exist
        missing = [w for w in [a, b, c] if w not in self._vocab[year]]
        if missing:
            if verbose:
                print(f"Words not found in {year}: {missing}")
            return []

        # Get vectors
        vec_a = self.get_vector(a, year)
        vec_b = self.get_vector(b, year)
        vec_c = self.get_vector(c, year)

        # Compute analogy vector: b - a + c
        analogy_vec = vec_b - vec_a + vec_c
        analogy_vec = analogy_vec / np.linalg.norm(analogy_vec)

        # Find most similar words
        similarities = np.dot(self._embeddings[year], analogy_vec)
        top_indices = np.argsort(similarities)[::-1]

        # Filter out input words
        input_words = {a, b, c}
        results = []
        for idx in top_indices:
            word = self._index_to_word[year][idx]
            if word not in input_words:
                results.append((word, float(similarities[idx])))
            if len(results) >= top_k:
                break

        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALOGY: '{a}' is to '{b}' as '{c}' is to ? ({year})")
            print(f"{'='*60}")
            for i, (w, sim) in enumerate(results, 1):
                print(f"  {i}. {w:20} {sim:.4f}")

        return results

    def analogy_over_time(self, a: str, b: str, c: str, top_k: int = 3) -> Dict:
        """
        Run an analogy across all time periods.

        Args:
            a, b, c: Words for the analogy
            top_k: Number of results per year

        Returns:
            Dictionary with results per year
        """
        print(f"\n{'='*60}")
        print(f"ANALOGY OVER TIME: '{a}' is to '{b}' as '{c}' is to ?")
        print(f"{'='*60}")

        results = {}
        for year in self.years:
            answers = self.analogy(a, b, c, year, top_k=top_k, verbose=False)
            if answers:
                results[year] = answers
                top_answer = answers[0][0]
                print(f"  {year}: {top_answer:15} (also: {', '.join([w for w,_ in answers[1:]])})")

        # Analyze consistency
        if results:
            all_top_answers = [r[0][0] for r in results.values()]
            from collections import Counter
            answer_counts = Counter(all_top_answers)
            most_common = answer_counts.most_common(3)

            print(f"\nMost frequent answers:")
            for answer, count in most_common:
                print(f"  {answer}: {count} times")

        return results

    def gender_analogy_test(self, year: int) -> Dict:
        """
        Test common gender analogies in a given year.

        Args:
            year: The year to test

        Returns:
            Results of gender analogy tests
        """
        print(f"\n{'='*60}")
        print(f"GENDER ANALOGY TEST ({year})")
        print(f"{'='*60}")

        analogies = [
            ("man", "king", "woman"),      # Expected: queen
            ("man", "doctor", "woman"),    # Shows historical bias
            ("man", "programmer", "woman"),
            ("he", "actor", "she"),        # Expected: actress
            ("man", "father", "woman"),    # Expected: mother
        ]

        results = {}
        for a, b, c in analogies:
            answers = self.analogy(a, b, c, year, top_k=3, verbose=False)
            if answers:
                results[f"{a}:{b}::{c}:?"] = answers
                print(f"\n  {a}:{b} :: {c}:?")
                for w, s in answers:
                    print(f"    -> {w} ({s:.3f})")

        return results

    # ==================== UTILITY METHODS ====================

    def vocabulary_stats(self, year: Optional[int] = None) -> Dict:
        """Get vocabulary statistics for a year or all years."""
        if year is not None:
            self._load_year(year)
            vocab_size = len(self._vocab[year])
            embed_shape = self._embeddings[year].shape
            return {
                "year": year,
                "vocabulary_size": vocab_size,
                "embedding_dimensions": embed_shape[1],
                "sample_words": list(self._vocab[year].keys())[:10]
            }

        stats = {}
        for y in self.years:
            stats[y] = self.vocabulary_stats(y)
        return stats

    def find_words_by_pattern(self, pattern: str, year: int) -> List[str]:
        """Find words matching a pattern (simple substring match)."""
        self._load_year(year)
        return [w for w in self._vocab[year].keys() if pattern in w]

    # ==================== VISUALIZATION METHODS ====================

    def plot_semantic_change(self, word: str, save_path: Optional[str] = None,
                              save: bool = True) -> Optional[Dict]:
        """
        Plot semantic change of a word over time.

        Args:
            word: The word to analyze
            save_path: Optional path to save the figure (defaults to output folder)
            save: Whether to save the figure (default True)

        Returns:
            Dictionary with change data, or None if plotting unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return None

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path(f"{word}_semantic_change.png")

        # Find years where word exists
        available_years = []
        for year in self.years:
            self._load_year(year)
            if word in self._vocab[year]:
                available_years.append(year)

        if len(available_years) < 2:
            print(f"Word '{word}' found in fewer than 2 time periods.")
            return None

        # Get vectors and compute changes
        vectors = {y: self.get_vector(word, y) for y in available_years}

        # Compute similarity with first year (drift from origin)
        first_year = available_years[0]
        drift_from_origin = []
        for year in available_years:
            sim = float(np.dot(vectors[first_year], vectors[year]))
            drift_from_origin.append(1 - sim)

        # Compute consecutive changes
        consecutive_changes = [0]  # First year has no change
        for i in range(1, len(available_years)):
            y1, y2 = available_years[i-1], available_years[i]
            sim = float(np.dot(vectors[y1], vectors[y2]))
            consecutive_changes.append(1 - sim)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Cumulative drift from first year
        axes[0].plot(available_years, drift_from_origin, 'b-o', linewidth=2, markersize=8)
        axes[0].fill_between(available_years, drift_from_origin, alpha=0.3)
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Semantic Distance from Origin', fontsize=12)
        axes[0].set_title(f"Cumulative Semantic Drift: '{word}'\n(Distance from {first_year})", fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        # Plot 2: Consecutive changes (rate of change)
        colors = ['green' if c < 0.05 else 'orange' if c < 0.1 else 'red' for c in consecutive_changes]
        axes[1].bar(available_years, consecutive_changes, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Change from Previous Period', fontsize=12)
        axes[1].set_title(f"Rate of Semantic Change: '{word}'", fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return {
            "word": word,
            "years": available_years,
            "drift_from_origin": drift_from_origin,
            "consecutive_changes": consecutive_changes
        }

    def plot_similarity_over_time(self, word_pairs: List[Tuple[str, str]],
                                   save_path: Optional[str] = None,
                                   save: bool = True) -> Optional[Dict]:
        """
        Plot similarity between word pairs over time.

        Args:
            word_pairs: List of (word1, word2) tuples to compare
            save_path: Optional path to save the figure (defaults to output folder)
            save: Whether to save the figure (default True)

        Returns:
            Dictionary with similarity data
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return None

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path("word_pair_similarity.png")

        fig, ax = plt.subplots(figsize=(12, 6))

        results = {}
        for word1, word2 in word_pairs:
            years_data = []
            sims = []

            for year in self.years:
                vec1 = self.get_vector(word1, year)
                vec2 = self.get_vector(word2, year)

                if vec1 is not None and vec2 is not None:
                    sim = float(np.dot(vec1, vec2))
                    years_data.append(year)
                    sims.append(sim)

            if years_data:
                ax.plot(years_data, sims, '-o', label=f"'{word1}' vs '{word2}'",
                       linewidth=2, markersize=6)
                results[f"{word1}-{word2}"] = {"years": years_data, "similarities": sims}

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Word Pair Similarity Over Time', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return results

    def plot_neighborhood_evolution(self, word: str, top_k: int = 10,
                                     years: Optional[List[int]] = None,
                                     save_path: Optional[str] = None,
                                     save: bool = True) -> Optional[Dict]:
        """
        Visualize how a word's neighborhood evolves over time.

        Args:
            word: The word to analyze
            top_k: Number of neighbors to track
            years: Specific years to show (defaults to all)
            save_path: Optional path to save the figure (defaults to output folder)
            save: Whether to save the figure (default True)

        Returns:
            Dictionary with neighborhood data
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return None

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path(f"{word}_neighborhood.png")

        if years is None:
            years = self.years

        # Collect all neighbors across all years
        all_neighbors = {}
        neighbor_data = {}

        for year in years:
            neighbors = self.similar_words(word, year, top_k=top_k, verbose=False)
            if neighbors:
                neighbor_data[year] = {w: s for w, s in neighbors}
                for w, s in neighbors:
                    if w not in all_neighbors:
                        all_neighbors[w] = {}
                    all_neighbors[w][year] = s

        if not neighbor_data:
            print(f"No data found for '{word}'")
            return None

        # Find most persistent neighbors (appear in most years)
        neighbor_persistence = {w: len(years_dict) for w, years_dict in all_neighbors.items()}
        top_persistent = sorted(neighbor_persistence.items(), key=lambda x: -x[1])[:top_k]
        words_to_plot = [w for w, _ in top_persistent]

        # Create heatmap data
        heatmap_data = []
        for w in words_to_plot:
            row = []
            for year in years:
                if year in all_neighbors.get(w, {}):
                    row.append(all_neighbors[w][year])
                else:
                    row.append(0)
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.set_yticks(range(len(words_to_plot)))
        ax.set_yticklabels(words_to_plot)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Neighbor Words', fontsize=12)
        ax.set_title(f"Neighborhood Evolution: '{word}'\n(Similarity shown by color intensity)", fontsize=14)

        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return {
            "word": word,
            "years": years,
            "top_neighbors": words_to_plot,
            "neighbor_data": neighbor_data
        }

    # ==================== WORD BIRTH/DEATH DETECTION ====================

    def detect_word_births(self, min_persistence: int = 3) -> Dict[int, List[str]]:
        """
        Detect when new words enter the vocabulary.

        Args:
            min_persistence: Minimum number of consecutive years a word must appear

        Returns:
            Dictionary mapping birth year to list of words born that year
        """
        print(f"\n{'='*60}")
        print("WORD BIRTH DETECTION")
        print(f"{'='*60}")

        births = {}
        seen_words = set()

        for i, year in enumerate(self.years):
            self._load_year(year)
            current_vocab = set(self._vocab[year].keys())

            if i == 0:
                seen_words = current_vocab
                continue

            # Find new words
            new_words = current_vocab - seen_words

            # Filter by persistence (check if they stay in vocabulary)
            persistent_new = []
            for word in new_words:
                persistence = 1
                for future_year in self.years[i+1:i+min_persistence]:
                    self._load_year(future_year)
                    if word in self._vocab[future_year]:
                        persistence += 1
                    else:
                        break

                if persistence >= min_persistence or i + min_persistence > len(self.years):
                    persistent_new.append(word)

            if persistent_new:
                births[year] = sorted(persistent_new)

            seen_words.update(current_vocab)

        # Print summary
        print(f"\nWords entering vocabulary by decade:")
        print("-" * 40)
        for year in sorted(births.keys()):
            count = len(births[year])
            sample = births[year][:5]
            print(f"  {year}: {count} new words (e.g., {', '.join(sample)}...)")

        return births

    def detect_word_deaths(self, min_absence: int = 3) -> Dict[int, List[str]]:
        """
        Detect when words leave the vocabulary.

        Args:
            min_absence: Minimum number of consecutive years a word must be absent

        Returns:
            Dictionary mapping death year to list of words that died
        """
        print(f"\n{'='*60}")
        print("WORD DEATH DETECTION")
        print(f"{'='*60}")

        deaths = {}

        # Track last appearance of each word
        last_seen = {}

        for year in self.years:
            self._load_year(year)
            for word in self._vocab[year].keys():
                last_seen[word] = year

        # Find words that stopped appearing before the end
        final_year = self.years[-1]

        for word, last_year in last_seen.items():
            if last_year < final_year:
                # Check if it stays absent
                last_idx = self.years.index(last_year)
                years_absent = len(self.years) - last_idx - 1

                if years_absent >= min_absence:
                    death_year = self.years[last_idx + 1]  # Year after last seen
                    if death_year not in deaths:
                        deaths[death_year] = []
                    deaths[death_year].append(word)

        # Print summary
        print(f"\nWords leaving vocabulary by decade:")
        print("-" * 40)
        for year in sorted(deaths.keys()):
            deaths[year] = sorted(deaths[year])
            count = len(deaths[year])
            sample = deaths[year][:5]
            print(f"  {year}: {count} words died (e.g., {', '.join(sample)}...)")

        return deaths

    def word_lifespan(self, word: str) -> Dict:
        """
        Analyze the lifespan of a word in the corpus.

        Args:
            word: The word to analyze

        Returns:
            Dictionary with birth, death, and lifespan information
        """
        existence = self.word_exists(word)

        years_present = [y for y, exists in existence.items() if exists]

        if not years_present:
            print(f"Word '{word}' not found in any year")
            return {"word": word, "found": False}

        birth = min(years_present)
        death = max(years_present)
        gaps = []

        # Find gaps in existence
        for i in range(len(years_present) - 1):
            gap = years_present[i + 1] - years_present[i]
            if gap > (self.years[1] - self.years[0]):  # More than one period
                gaps.append((years_present[i], years_present[i + 1]))

        result = {
            "word": word,
            "found": True,
            "first_appearance": birth,
            "last_appearance": death,
            "lifespan_years": death - birth,
            "periods_present": len(years_present),
            "gaps": gaps
        }

        print(f"\n{'='*60}")
        print(f"WORD LIFESPAN: '{word}'")
        print(f"{'='*60}")
        print(f"  First appearance: {birth}")
        print(f"  Last appearance: {death}")
        print(f"  Lifespan: {death - birth} years")
        print(f"  Periods present: {len(years_present)} of {len(self.years)}")
        if gaps:
            print(f"  Gaps in existence: {gaps}")

        return result

    def find_neologisms(self, start_year: int, end_year: Optional[int] = None,
                        top_k: int = 50) -> List[Tuple[str, int]]:
        """
        Find words that appeared between start_year and end_year.

        Args:
            start_year: Beginning of the period to search
            end_year: End of the period (defaults to latest year)
            top_k: Number of neologisms to return

        Returns:
            List of (word, birth_year) tuples
        """
        if end_year is None:
            end_year = self.years[-1]

        print(f"\n{'='*60}")
        print(f"NEOLOGISMS: {start_year} - {end_year}")
        print(f"{'='*60}")

        # Get vocabulary before start_year
        pre_vocab = set()
        for year in self.years:
            if year < start_year:
                self._load_year(year)
                pre_vocab.update(self._vocab[year].keys())

        # Find new words in the period
        neologisms = []
        for year in self.years:
            if start_year <= year <= end_year:
                self._load_year(year)
                new_words = set(self._vocab[year].keys()) - pre_vocab
                for word in new_words:
                    neologisms.append((word, year))
                pre_vocab.update(self._vocab[year].keys())

        # Sort by year, then alphabetically
        neologisms.sort(key=lambda x: (x[1], x[0]))

        print(f"\nFound {len(neologisms)} new words in this period")
        print(f"Top {top_k} neologisms:")
        for word, year in neologisms[:top_k]:
            print(f"  {year}: {word}")

        return neologisms[:top_k]

    # ==================== SEMANTIC DRIFT CLUSTERING ====================

    def compute_change_vectors(self, words: Optional[List[str]] = None,
                                sample_size: int = 1000) -> Tuple[List[str], np.ndarray]:
        """
        Compute semantic change vectors for words.

        Args:
            words: List of words to analyze (if None, samples from vocabulary)
            sample_size: Number of words to sample if words is None

        Returns:
            Tuple of (word list, change vector matrix)
        """
        if len(self.years) < 2:
            raise ValueError("Need at least 2 time periods")

        first_year = self.years[0]
        last_year = self.years[-1]

        self._load_year(first_year)
        self._load_year(last_year)

        # Find words present in both periods
        if words is None:
            common_vocab = set(self._vocab[first_year].keys()) & set(self._vocab[last_year].keys())
            words = list(common_vocab)[:sample_size]
        else:
            words = [w for w in words if w in self._vocab[first_year] and w in self._vocab[last_year]]

        # Compute change vectors (difference between final and initial embeddings)
        change_vectors = []
        valid_words = []

        for word in words:
            vec_first = self.get_vector(word, first_year)
            vec_last = self.get_vector(word, last_year)

            if vec_first is not None and vec_last is not None:
                change_vec = vec_last - vec_first
                change_vectors.append(change_vec)
                valid_words.append(word)

        return valid_words, np.array(change_vectors)

    def cluster_semantic_drift(self, n_clusters: int = 5,
                                sample_size: int = 500,
                                words: Optional[List[str]] = None) -> Dict:
        """
        Cluster words by their semantic drift patterns.

        Args:
            n_clusters: Number of clusters to create
            sample_size: Number of words to analyze
            words: Optional list of specific words to cluster

        Returns:
            Dictionary with cluster assignments and analysis
        """
        if not HAS_SKLEARN:
            print("Scikit-learn not installed. Install with: pip install scikit-learn")
            return {}

        print(f"\n{'='*60}")
        print("SEMANTIC DRIFT CLUSTERING")
        print(f"{'='*60}")

        valid_words, change_vectors = self.compute_change_vectors(words, sample_size)

        if len(valid_words) < n_clusters:
            print(f"Not enough words ({len(valid_words)}) for {n_clusters} clusters")
            return {}

        print(f"Clustering {len(valid_words)} words into {n_clusters} groups...")

        # Normalize change vectors
        scaler = StandardScaler()
        change_vectors_scaled = scaler.fit_transform(change_vectors)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(change_vectors_scaled)

        # Analyze clusters
        clusters = {i: [] for i in range(n_clusters)}
        for word, label in zip(valid_words, labels):
            clusters[label].append(word)

        # Compute cluster statistics
        first_year = self.years[0]
        last_year = self.years[-1]

        cluster_stats = {}
        for cluster_id, cluster_words in clusters.items():
            # Average change magnitude for cluster
            changes = []
            for word in cluster_words:
                vec_first = self.get_vector(word, first_year)
                vec_last = self.get_vector(word, last_year)
                sim = float(np.dot(vec_first, vec_last))
                changes.append(1 - sim)

            avg_change = np.mean(changes)
            cluster_stats[cluster_id] = {
                "size": len(cluster_words),
                "avg_change": avg_change,
                "sample_words": cluster_words[:10]
            }

        # Print results
        print(f"\nCluster Analysis ({first_year} → {last_year}):")
        print("-" * 50)
        for cluster_id in sorted(cluster_stats.keys(), key=lambda x: -cluster_stats[x]["avg_change"]):
            stats = cluster_stats[cluster_id]
            print(f"\nCluster {cluster_id}: {stats['size']} words, avg change: {stats['avg_change']:.4f}")
            print(f"  Sample: {', '.join(stats['sample_words'][:8])}")

        return {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "cluster_stats": cluster_stats,
            "period": (first_year, last_year)
        }

    def find_co_evolving_words(self, target_word: str, top_k: int = 20,
                                sample_size: int = 1000) -> List[Tuple[str, float]]:
        """
        Find words that changed in similar ways to the target word.

        Args:
            target_word: The word to find co-evolving partners for
            top_k: Number of similar words to return
            sample_size: Number of words to compare against

        Returns:
            List of (word, similarity) tuples
        """
        print(f"\n{'='*60}")
        print(f"CO-EVOLVING WORDS: '{target_word}'")
        print(f"{'='*60}")

        valid_words, change_vectors = self.compute_change_vectors(sample_size=sample_size)

        if target_word not in valid_words:
            print(f"Word '{target_word}' not found in analysis period")
            return []

        target_idx = valid_words.index(target_word)
        target_change = change_vectors[target_idx]
        target_change_norm = target_change / np.linalg.norm(target_change)

        # Compute similarity of change vectors
        similarities = []
        for i, (word, change_vec) in enumerate(zip(valid_words, change_vectors)):
            if i != target_idx:
                change_norm = change_vec / np.linalg.norm(change_vec)
                sim = float(np.dot(target_change_norm, change_norm))
                similarities.append((word, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        top_matches = similarities[:top_k]

        print(f"\nWords that changed similarly to '{target_word}':")
        print("-" * 40)
        for word, sim in top_matches:
            print(f"  {word:20} {sim:.4f}")

        return top_matches

    def plot_drift_clusters(self, n_clusters: int = 5, sample_size: int = 300,
                            save_path: Optional[str] = None,
                            save: bool = True) -> Optional[Dict]:
        """
        Visualize semantic drift clusters using PCA.

        Args:
            n_clusters: Number of clusters
            sample_size: Number of words to analyze
            save_path: Optional path to save figure (defaults to output folder)
            save: Whether to save the figure (default True)

        Returns:
            Cluster data dictionary
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            print("Requires matplotlib and scikit-learn")
            return None

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path("drift_clusters.png")

        valid_words, change_vectors = self.compute_change_vectors(sample_size=sample_size)

        if len(valid_words) < n_clusters:
            print(f"Not enough words for clustering")
            return None

        # Cluster
        scaler = StandardScaler()
        change_vectors_scaled = scaler.fit_transform(change_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(change_vectors_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(change_vectors_scaled)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                            cmap='tab10', alpha=0.6, s=50)

        # Label some points
        for i in range(0, len(valid_words), max(1, len(valid_words) // 30)):
            ax.annotate(valid_words[i], (coords[i, 0], coords[i, 1]),
                       fontsize=8, alpha=0.7)

        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'Semantic Drift Clusters\n({self.years[0]} → {self.years[-1]})', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Cluster')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return {"words": valid_words, "labels": labels.tolist(), "coords": coords.tolist()}

    # ==================== CULTURAL SHIFT DETECTION ====================

    def detect_cultural_shifts(self, top_k: int = 20,
                                sample_size: int = 2000) -> Dict[Tuple[int, int], List[Tuple[str, float]]]:
        """
        Automatically identify words with the largest semantic shifts per decade.

        Args:
            top_k: Number of top changing words per period
            sample_size: Number of words to analyze

        Returns:
            Dictionary mapping (year1, year2) to list of (word, change) tuples
        """
        print(f"\n{'='*60}")
        print("CULTURAL SHIFT DETECTION")
        print(f"{'='*60}")

        # Sample common vocabulary across all years
        common_vocab = None
        for year in self.years:
            self._load_year(year)
            if common_vocab is None:
                common_vocab = set(self._vocab[year].keys())
            else:
                common_vocab &= set(self._vocab[year].keys())

        words = list(common_vocab)[:sample_size]
        print(f"Analyzing {len(words)} words across {len(self.years)} time periods...")

        results = {}

        for i in range(len(self.years) - 1):
            year1, year2 = self.years[i], self.years[i + 1]

            changes = []
            for word in words:
                vec1 = self.get_vector(word, year1)
                vec2 = self.get_vector(word, year2)

                if vec1 is not None and vec2 is not None:
                    sim = float(np.dot(vec1, vec2))
                    change = 1 - sim
                    changes.append((word, change))

            # Sort by change magnitude
            changes.sort(key=lambda x: -x[1])
            results[(year1, year2)] = changes[:top_k]

            print(f"\n{year1} → {year2}: Top changing words")
            print("-" * 40)
            for word, change in changes[:10]:
                print(f"  {word:20} {change:.4f}")

        return results

    def identify_shift_periods(self, word: str) -> Dict:
        """
        Identify periods of significant semantic shift for a word.

        Args:
            word: The word to analyze

        Returns:
            Dictionary with shift analysis
        """
        print(f"\n{'='*60}")
        print(f"SHIFT PERIOD IDENTIFICATION: '{word}'")
        print(f"{'='*60}")

        # Collect change data
        changes = []
        for i in range(len(self.years) - 1):
            year1, year2 = self.years[i], self.years[i + 1]

            vec1 = self.get_vector(word, year1)
            vec2 = self.get_vector(word, year2)

            if vec1 is not None and vec2 is not None:
                sim = float(np.dot(vec1, vec2))
                changes.append({
                    "from_year": year1,
                    "to_year": year2,
                    "change": 1 - sim
                })

        if not changes:
            print(f"Word '{word}' not found in consecutive periods")
            return {}

        # Calculate statistics
        change_values = [c["change"] for c in changes]
        mean_change = np.mean(change_values)
        std_change = np.std(change_values)

        # Identify significant shifts (> 1.5 std above mean)
        threshold = mean_change + 1.5 * std_change
        significant_shifts = [c for c in changes if c["change"] > threshold]

        print(f"\nChange statistics:")
        print(f"  Mean change per period: {mean_change:.4f}")
        print(f"  Std deviation: {std_change:.4f}")
        print(f"  Significance threshold: {threshold:.4f}")

        print(f"\nSignificant shift periods:")
        if significant_shifts:
            for shift in significant_shifts:
                print(f"  {shift['from_year']} → {shift['to_year']}: {shift['change']:.4f}")
        else:
            print("  No periods with unusually high change detected")

        # Find most stable period
        min_change = min(changes, key=lambda x: x["change"])
        print(f"\nMost stable period:")
        print(f"  {min_change['from_year']} → {min_change['to_year']}: {min_change['change']:.4f}")

        return {
            "word": word,
            "changes": changes,
            "mean_change": mean_change,
            "std_change": std_change,
            "significant_shifts": significant_shifts,
            "most_stable": min_change
        }

    def cultural_era_analysis(self, era_name: str, start_year: int, end_year: int,
                               keywords: List[str], top_k: int = 10) -> Dict:
        """
        Analyze semantic changes within a cultural era.

        Args:
            era_name: Name for this era (e.g., "Industrial Revolution")
            start_year: Start of the era
            end_year: End of the era
            keywords: Seed words relevant to this era
            top_k: Number of related words to find

        Returns:
            Era analysis results
        """
        print(f"\n{'='*60}")
        print(f"CULTURAL ERA ANALYSIS: {era_name}")
        print(f"Period: {start_year} - {end_year}")
        print(f"{'='*60}")

        # Find closest years in our dataset
        era_years = [y for y in self.years if start_year <= y <= end_year]
        if not era_years:
            print(f"No data available for this period")
            return {}

        results = {
            "era_name": era_name,
            "period": (start_year, end_year),
            "available_years": era_years,
            "keyword_analysis": {}
        }

        for keyword in keywords:
            print(f"\n--- Analyzing: '{keyword}' ---")

            # Check if word exists in era
            existence = {y: self.word_exists(keyword, y) for y in era_years}
            years_present = [y for y, exists in existence.items() if exists]

            if len(years_present) < 2:
                print(f"  Word not found in enough periods")
                continue

            # Semantic change within era
            first, last = min(years_present), max(years_present)
            vec_first = self.get_vector(keyword, first)
            vec_last = self.get_vector(keyword, last)

            if vec_first is not None and vec_last is not None:
                era_change = 1 - float(np.dot(vec_first, vec_last))
                print(f"  Semantic change ({first}→{last}): {era_change:.4f}")

            # Find related words that emerged or changed
            if first in years_present:
                neighbors_start = self.similar_words(keyword, first, top_k=top_k, verbose=False)
                neighbors_end = self.similar_words(keyword, last, top_k=top_k, verbose=False)

                start_set = set([w for w, _ in neighbors_start])
                end_set = set([w for w, _ in neighbors_end])

                emerged = end_set - start_set
                departed = start_set - end_set

                print(f"  Neighbors gained: {list(emerged)[:5]}")
                print(f"  Neighbors lost: {list(departed)[:5]}")

                results["keyword_analysis"][keyword] = {
                    "era_change": era_change,
                    "years_present": years_present,
                    "neighbors_gained": list(emerged),
                    "neighbors_lost": list(departed)
                }

        return results

    def plot_cultural_shifts(self, top_k: int = 10, sample_size: int = 1000,
                              save_path: Optional[str] = None,
                              save: bool = True) -> Optional[Dict]:
        """
        Visualize cultural shifts across all time periods.

        Args:
            top_k: Number of top changing words to show
            sample_size: Number of words to analyze
            save_path: Optional path to save figure (defaults to output folder)
            save: Whether to save the figure (default True)

        Returns:
            Shift data dictionary
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed")
            return None

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path("cultural_shifts.png")

        shifts = self.detect_cultural_shifts(top_k=top_k, sample_size=sample_size)

        # Create visualization
        periods = sorted(shifts.keys())

        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate average change per period
        avg_changes = []
        for period in periods:
            avg = np.mean([c for _, c in shifts[period]])
            avg_changes.append(avg)

        x_labels = [f"{p[0]}-{p[1]}" for p in periods]
        x_pos = range(len(periods))

        bars = ax.bar(x_pos, avg_changes, color='steelblue', alpha=0.7, edgecolor='black')

        # Highlight period with most change
        max_idx = np.argmax(avg_changes)
        bars[max_idx].set_color('crimson')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Average Semantic Change', fontsize=12)
        ax.set_title('Cultural Shift Intensity by Period', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Add annotations for top changing words in highlighted period
        top_words = [w for w, _ in shifts[periods[max_idx]][:3]]
        ax.annotate(f"Top: {', '.join(top_words)}",
                   xy=(max_idx, avg_changes[max_idx]),
                   xytext=(max_idx, avg_changes[max_idx] + 0.02),
                   ha='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return {"periods": periods, "avg_changes": avg_changes, "shifts": shifts}


def list_available_datasets(base_dir: Optional[str] = None) -> List[str]:
    """List all available HistWords datasets in the directory."""
    if base_dir is None:
        base_dir = Path(__file__).parent

    base_dir = Path(base_dir)
    datasets = []

    for d in base_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            # Check if it contains vocab files
            if list(d.glob("*-vocab.pkl")):
                datasets.append(d.name)

    return sorted(datasets)


# ==================== DEMO / MAIN ====================

if __name__ == "__main__":
    print("HistWords Analysis Toolkit")
    print("=" * 60)

    # List available datasets
    datasets = list_available_datasets()
    print(f"\nAvailable datasets: {datasets}")

    # Demo with English data
    if "eng-all_sgns" in datasets:
        print("\n" + "=" * 60)
        print("DEMO: English Google N-grams")
        print("=" * 60)

        analyzer = HistWordsAnalyzer("eng-all_sgns")

        # Vocabulary stats
        print("\nVocabulary stats for 1900:")
        stats = analyzer.vocabulary_stats(1900)
        print(f"  Size: {stats['vocabulary_size']} words")
        print(f"  Dimensions: {stats['embedding_dimensions']}")

        # Semantic change example
        analyzer.semantic_change("computer", method="cosine")

        # Similar words
        analyzer.similar_words("woman", year=1900, top_k=10)
        analyzer.similar_words("woman", year=1990, top_k=10)

        # Analogy
        analyzer.analogy("king", "man", "queen", year=1950)

        # Similarity over time
        analyzer.similarity_over_time("gay", "happy")

        print("\n" + "=" * 60)
        print("Demo complete! Import this module to use in your own analysis.")
