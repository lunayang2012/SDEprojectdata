"""
Descriptive Analysis of HistWords Embeddings
=============================================

This script provides basic descriptive statistics and summaries
of the historical word embeddings datasets.

Usage:
    python descriptive_analysis.py
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple
from histwords_analysis import HistWordsAnalyzer, list_available_datasets, get_output_path

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


class DescriptiveAnalyzer:
    """Perform descriptive analysis on HistWords embeddings."""

    def __init__(self, analyzer: HistWordsAnalyzer):
        """
        Initialize with a HistWordsAnalyzer instance.

        Args:
            analyzer: Loaded HistWordsAnalyzer object
        """
        self.analyzer = analyzer
        self.dataset_name = analyzer.dataset_name
        self.years = analyzer.years

    def dataset_overview(self) -> Dict:
        """
        Generate overview statistics for the dataset.

        Returns:
            Dictionary with overview statistics
        """
        print_header(f"DATASET OVERVIEW: {self.dataset_name}")

        overview = {
            "dataset_name": self.dataset_name,
            "num_time_periods": len(self.years),
            "year_range": (min(self.years), max(self.years)),
            "time_span_years": max(self.years) - min(self.years),
            "decade_interval": self.years[1] - self.years[0] if len(self.years) > 1 else 0
        }

        print(f"  Dataset: {overview['dataset_name']}")
        print(f"  Time periods: {overview['num_time_periods']}")
        print(f"  Year range: {overview['year_range'][0]} - {overview['year_range'][1]}")
        print(f"  Total span: {overview['time_span_years']} years")
        print(f"  Interval: {overview['decade_interval']} years between periods")

        return overview

    def vocabulary_statistics(self) -> Dict:
        """
        Compute vocabulary statistics across all time periods.

        Returns:
            Dictionary with vocabulary stats per year
        """
        print_header("VOCABULARY STATISTICS")

        stats = {}
        vocab_sizes = []
        embedding_dims = []

        for year in self.years:
            self.analyzer._load_year(year)
            vocab_size = len(self.analyzer._vocab[year])
            embed_dim = self.analyzer._embeddings[year].shape[1]

            stats[year] = {
                "vocab_size": vocab_size,
                "embedding_dim": embed_dim
            }
            vocab_sizes.append(vocab_size)
            embedding_dims.append(embed_dim)

        # Summary statistics
        summary = {
            "min_vocab": min(vocab_sizes),
            "max_vocab": max(vocab_sizes),
            "mean_vocab": np.mean(vocab_sizes),
            "std_vocab": np.std(vocab_sizes),
            "embedding_dim": embedding_dims[0],  # Usually constant
            "year_stats": stats
        }

        print(f"\n  Vocabulary Size:")
        print(f"    Minimum: {summary['min_vocab']:,} words")
        print(f"    Maximum: {summary['max_vocab']:,} words")
        print(f"    Mean: {summary['mean_vocab']:,.0f} words")
        print(f"    Std Dev: {summary['std_vocab']:,.0f} words")
        print(f"\n  Embedding Dimension: {summary['embedding_dim']}")

        print_subheader("Vocabulary by Year")
        for year in self.years:
            print(f"    {year}: {stats[year]['vocab_size']:,} words")

        return summary

    def vocabulary_overlap_analysis(self) -> Dict:
        """
        Analyze vocabulary overlap between consecutive periods and overall.

        Returns:
            Dictionary with overlap statistics
        """
        print_header("VOCABULARY OVERLAP ANALYSIS")

        # Load all vocabularies
        vocabs = {}
        for year in self.years:
            self.analyzer._load_year(year)
            vocabs[year] = set(self.analyzer._vocab[year].keys())

        # Core vocabulary (present in ALL periods)
        core_vocab = set.intersection(*vocabs.values())
        print(f"\n  Core vocabulary (in all periods): {len(core_vocab):,} words")

        # Sample core words
        core_sample = sorted(list(core_vocab))[:20]
        print(f"  Sample core words: {', '.join(core_sample)}")

        # Union vocabulary (present in ANY period)
        union_vocab = set.union(*vocabs.values())
        print(f"\n  Total unique words (any period): {len(union_vocab):,} words")

        # Consecutive overlap
        print_subheader("Consecutive Period Overlap")
        overlaps = []
        for i in range(len(self.years) - 1):
            y1, y2 = self.years[i], self.years[i + 1]
            overlap = len(vocabs[y1] & vocabs[y2])
            union = len(vocabs[y1] | vocabs[y2])
            jaccard = overlap / union if union > 0 else 0
            overlaps.append({
                "from_year": y1,
                "to_year": y2,
                "overlap_count": overlap,
                "jaccard_similarity": jaccard
            })
            print(f"    {y1} -> {y2}: {overlap:,} shared words (Jaccard: {jaccard:.4f})")

        # First to last comparison
        first_year, last_year = self.years[0], self.years[-1]
        first_last_overlap = len(vocabs[first_year] & vocabs[last_year])
        first_last_jaccard = first_last_overlap / len(vocabs[first_year] | vocabs[last_year])

        print(f"\n  First to Last ({first_year} -> {last_year}):")
        print(f"    Shared words: {first_last_overlap:,}")
        print(f"    Jaccard similarity: {first_last_jaccard:.4f}")

        return {
            "core_vocab_size": len(core_vocab),
            "core_vocab_sample": core_sample,
            "union_vocab_size": len(union_vocab),
            "consecutive_overlaps": overlaps,
            "first_last_overlap": first_last_overlap,
            "first_last_jaccard": first_last_jaccard
        }

    def semantic_change_distribution(self, sample_size: int = 1000) -> Dict:
        """
        Analyze the distribution of semantic change across words.

        Args:
            sample_size: Number of words to sample for analysis

        Returns:
            Dictionary with distribution statistics
        """
        print_header("SEMANTIC CHANGE DISTRIBUTION")

        # Get words present in both first and last year
        first_year, last_year = self.years[0], self.years[-1]
        self.analyzer._load_year(first_year)
        self.analyzer._load_year(last_year)

        common_words = list(
            set(self.analyzer._vocab[first_year].keys()) &
            set(self.analyzer._vocab[last_year].keys())
        )

        # Sample if necessary
        if len(common_words) > sample_size:
            np.random.seed(42)
            sample_words = np.random.choice(common_words, sample_size, replace=False)
        else:
            sample_words = common_words

        # Compute semantic change for each word
        changes = []
        for word in sample_words:
            vec_first = self.analyzer.get_vector(word, first_year)
            vec_last = self.analyzer.get_vector(word, last_year)
            if vec_first is not None and vec_last is not None:
                sim = float(np.dot(vec_first, vec_last))
                change = 1 - sim
                changes.append({"word": word, "change": change, "similarity": sim})

        # Sort by change
        changes.sort(key=lambda x: x["change"], reverse=True)

        # Statistics
        change_values = [c["change"] for c in changes]
        stats = {
            "num_words_analyzed": len(changes),
            "mean_change": np.mean(change_values),
            "median_change": np.median(change_values),
            "std_change": np.std(change_values),
            "min_change": np.min(change_values),
            "max_change": np.max(change_values),
            "percentile_25": np.percentile(change_values, 25),
            "percentile_75": np.percentile(change_values, 75),
            "percentile_90": np.percentile(change_values, 90),
            "percentile_95": np.percentile(change_values, 95)
        }

        print(f"\n  Analysis period: {first_year} -> {last_year}")
        print(f"  Words analyzed: {stats['num_words_analyzed']:,}")
        print(f"\n  Semantic Change Statistics:")
        print(f"    Mean: {stats['mean_change']:.4f}")
        print(f"    Median: {stats['median_change']:.4f}")
        print(f"    Std Dev: {stats['std_change']:.4f}")
        print(f"    Min: {stats['min_change']:.4f}")
        print(f"    Max: {stats['max_change']:.4f}")
        print(f"\n  Percentiles:")
        print(f"    25th: {stats['percentile_25']:.4f}")
        print(f"    75th: {stats['percentile_75']:.4f}")
        print(f"    90th: {stats['percentile_90']:.4f}")
        print(f"    95th: {stats['percentile_95']:.4f}")

        # Most and least changed words
        print_subheader("Top 10 Most Changed Words")
        for i, item in enumerate(changes[:10], 1):
            print(f"    {i:2}. {item['word']:20} change={item['change']:.4f}")

        print_subheader("Top 10 Least Changed Words")
        for i, item in enumerate(changes[-10:], 1):
            print(f"    {i:2}. {item['word']:20} change={item['change']:.4f}")

        stats["most_changed"] = changes[:20]
        stats["least_changed"] = changes[-20:]

        return stats

    def word_frequency_proxy(self, year: int, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Estimate word frequency/importance using vector norms.

        Note: This is a rough proxy - actual frequency data would be better.

        Args:
            year: Year to analyze
            top_k: Number of top words to return

        Returns:
            List of (word, norm) tuples
        """
        print_subheader(f"Word Importance Proxy ({year})")

        self.analyzer._load_year(year)
        vocab = self.analyzer._vocab[year]
        embeddings = self.analyzer._embeddings[year]

        # Note: embeddings are normalized, so we look at other properties
        # Here we just return sample vocabulary
        words = list(vocab.keys())[:top_k]

        print(f"  Sample vocabulary ({top_k} words): {', '.join(words[:20])}...")

        return [(w, 1.0) for w in words]

    def embedding_statistics(self, year: int) -> Dict:
        """
        Compute statistics about the embedding vectors.

        Args:
            year: Year to analyze

        Returns:
            Dictionary with embedding statistics
        """
        print_subheader(f"Embedding Statistics ({year})")

        self.analyzer._load_year(year)
        embeddings = self.analyzer._embeddings[year]

        # Vector statistics
        norms = np.linalg.norm(embeddings, axis=1)
        means = np.mean(embeddings, axis=0)
        stds = np.std(embeddings, axis=0)

        stats = {
            "num_vectors": embeddings.shape[0],
            "dimensions": embeddings.shape[1],
            "norm_mean": np.mean(norms),
            "norm_std": np.std(norms),
            "value_mean": np.mean(embeddings),
            "value_std": np.std(embeddings),
            "value_min": np.min(embeddings),
            "value_max": np.max(embeddings)
        }

        print(f"    Vectors: {stats['num_vectors']:,}")
        print(f"    Dimensions: {stats['dimensions']}")
        print(f"    Norm - Mean: {stats['norm_mean']:.4f}, Std: {stats['norm_std']:.4f}")
        print(f"    Values - Mean: {stats['value_mean']:.4f}, Std: {stats['value_std']:.4f}")
        print(f"    Value range: [{stats['value_min']:.4f}, {stats['value_max']:.4f}]")

        return stats

    def word_length_analysis(self) -> Dict:
        """
        Analyze word lengths in the vocabulary.

        Returns:
            Dictionary with word length statistics
        """
        print_header("WORD LENGTH ANALYSIS")

        all_lengths = []
        length_by_year = {}

        for year in self.years:
            self.analyzer._load_year(year)
            words = list(self.analyzer._vocab[year].keys())
            lengths = [len(w) for w in words]
            all_lengths.extend(lengths)
            length_by_year[year] = {
                "mean": np.mean(lengths),
                "median": np.median(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }

        # Overall statistics
        stats = {
            "overall_mean": np.mean(all_lengths),
            "overall_median": np.median(all_lengths),
            "overall_min": min(all_lengths),
            "overall_max": max(all_lengths),
            "by_year": length_by_year
        }

        print(f"\n  Overall Word Length:")
        print(f"    Mean: {stats['overall_mean']:.2f} characters")
        print(f"    Median: {stats['overall_median']:.0f} characters")
        print(f"    Range: {stats['overall_min']} - {stats['overall_max']} characters")

        # Length distribution
        length_counts = Counter(all_lengths)
        print_subheader("Length Distribution (top 10)")
        for length, count in sorted(length_counts.items())[:15]:
            bar = "#" * min(50, count // 1000)
            print(f"    {length:2} chars: {count:6,} {bar}")

        return stats

    def analogy_accuracy_test(self, year: int = 1950) -> Dict:
        """
        Test analogy accuracy with known relationships.

        Args:
            year: Year to test

        Returns:
            Dictionary with accuracy results
        """
        print_header(f"ANALOGY ACCURACY TEST ({year})")

        # Test analogies (a:b :: c:expected)
        test_cases = [
            ("man", "king", "woman", "queen"),
            ("man", "father", "woman", "mother"),
            ("paris", "france", "london", "england"),
            ("paris", "france", "berlin", "germany"),
            ("good", "better", "bad", "worse"),
            ("slow", "slower", "fast", "faster"),
        ]

        results = []
        correct = 0

        for a, b, c, expected in test_cases:
            answers = self.analyzer.analogy(a, b, c, year, top_k=5, verbose=False)
            if answers:
                top_answer = answers[0][0]
                is_correct = top_answer == expected
                in_top5 = expected in [w for w, _ in answers]

                if is_correct:
                    correct += 1

                results.append({
                    "analogy": f"{a}:{b} :: {c}:?",
                    "expected": expected,
                    "predicted": top_answer,
                    "correct": is_correct,
                    "in_top5": in_top5
                })

                status = "✓" if is_correct else ("~" if in_top5 else "✗")
                print(f"  {status} {a}:{b} :: {c}:? -> {top_answer} (expected: {expected})")
            else:
                results.append({
                    "analogy": f"{a}:{b} :: {c}:?",
                    "expected": expected,
                    "predicted": None,
                    "correct": False,
                    "in_top5": False
                })
                print(f"  ✗ {a}:{b} :: {c}:? -> N/A (words not found)")

        accuracy = correct / len(test_cases) if test_cases else 0
        print(f"\n  Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")

        return {
            "year": year,
            "num_tests": len(test_cases),
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a complete summary report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        print_header("GENERATING SUMMARY REPORT")

        report_lines = [
            f"# HistWords Descriptive Analysis Report",
            f"## Dataset: {self.dataset_name}",
            f"",
            f"### Overview",
            f"- Time periods: {len(self.years)}",
            f"- Year range: {min(self.years)} - {max(self.years)}",
            f"- Total span: {max(self.years) - min(self.years)} years",
            f""
        ]

        # Add vocabulary stats
        vocab_stats = self.vocabulary_statistics()
        report_lines.extend([
            f"### Vocabulary Statistics",
            f"- Minimum vocabulary: {vocab_stats['min_vocab']:,} words",
            f"- Maximum vocabulary: {vocab_stats['max_vocab']:,} words",
            f"- Mean vocabulary: {vocab_stats['mean_vocab']:,.0f} words",
            f"- Embedding dimensions: {vocab_stats['embedding_dim']}",
            f""
        ])

        # Add semantic change stats
        change_stats = self.semantic_change_distribution(sample_size=500)
        report_lines.extend([
            f"### Semantic Change Distribution",
            f"- Mean change: {change_stats['mean_change']:.4f}",
            f"- Median change: {change_stats['median_change']:.4f}",
            f"- 90th percentile: {change_stats['percentile_90']:.4f}",
            f""
        ])

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"  Report saved to: {output_path}")

        return report

    def plot_vocabulary_over_time(self, save_path: Optional[str] = None, save: bool = True):
        """Plot vocabulary size over time."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed - skipping plot")
            return

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path("vocab_over_time.png")

        sizes = []
        for year in self.years:
            self.analyzer._load_year(year)
            sizes.append(len(self.analyzer._vocab[year]))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.years, sizes, 'b-o', linewidth=2, markersize=8)
        ax.fill_between(self.years, sizes, alpha=0.3)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Vocabulary Size', fontsize=12)
        ax.set_title(f'Vocabulary Size Over Time: {self.dataset_name}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        plt.show()

    def plot_semantic_change_histogram(self, sample_size: int = 1000,
                                        save_path: Optional[str] = None,
                                        save: bool = True):
        """Plot histogram of semantic change distribution."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed - skipping plot")
            return

        # Default save path to output directory
        if save and save_path is None:
            save_path = get_output_path("change_histogram.png")

        # Compute changes
        first_year, last_year = self.years[0], self.years[-1]
        self.analyzer._load_year(first_year)
        self.analyzer._load_year(last_year)

        common_words = list(
            set(self.analyzer._vocab[first_year].keys()) &
            set(self.analyzer._vocab[last_year].keys())
        )

        if len(common_words) > sample_size:
            np.random.seed(42)
            sample_words = np.random.choice(common_words, sample_size, replace=False)
        else:
            sample_words = common_words

        changes = []
        for word in sample_words:
            vec_first = self.analyzer.get_vector(word, first_year)
            vec_last = self.analyzer.get_vector(word, last_year)
            if vec_first is not None and vec_last is not None:
                sim = float(np.dot(vec_first, vec_last))
                changes.append(1 - sim)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(changes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(changes), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(changes):.3f}')
        ax.axvline(np.median(changes), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(changes):.3f}')
        ax.set_xlabel('Semantic Change (1 - cosine similarity)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of Semantic Change ({first_year} → {last_year})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        plt.show()


def main():
    """Run all descriptive analyses."""
    print_header("HISTWORDS DESCRIPTIVE ANALYSIS")

    # List available datasets
    print("\nAvailable datasets:")
    datasets = list_available_datasets()
    for ds in datasets:
        print(f"  - {ds}")

    # Load primary dataset
    print("\nLoading English dataset...")
    eng = HistWordsAnalyzer("eng-all_sgns")
    analyzer = DescriptiveAnalyzer(eng)

    # Run analyses
    analyzer.dataset_overview()
    analyzer.vocabulary_statistics()
    analyzer.vocabulary_overlap_analysis()
    analyzer.word_length_analysis()
    analyzer.semantic_change_distribution(sample_size=1000)

    # Embedding statistics for select years
    print_header("EMBEDDING STATISTICS BY YEAR")
    for year in [eng.years[0], eng.years[len(eng.years)//2], eng.years[-1]]:
        analyzer.embedding_statistics(year)

    # Analogy test
    analyzer.analogy_accuracy_test(year=1950)

    # Plots (if matplotlib available)
    if HAS_MATPLOTLIB:
        print_header("GENERATING PLOTS")
        analyzer.plot_vocabulary_over_time()
        analyzer.plot_semantic_change_histogram()

    print_header("ANALYSIS COMPLETE")
    print("""
Summary of analyses performed:
  1. Dataset Overview - time periods, year range
  2. Vocabulary Statistics - sizes, dimensions
  3. Vocabulary Overlap - core vocab, consecutive overlap
  4. Word Length Analysis - character distribution
  5. Semantic Change Distribution - change statistics
  6. Embedding Statistics - vector properties
  7. Analogy Accuracy Test - standard analogies

Plots generated (if matplotlib installed):
  - vocab_over_time.png
  - change_histogram.png
""")


if __name__ == "__main__":
    main()
