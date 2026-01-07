"""
Example Usage of HistWords Analysis Toolkit
===========================================

This script demonstrates the key features of the HistWords analysis toolkit.
Run each section to explore historical word embeddings.

Usage:
    python examples.py              # Run all examples
    python examples.py --quick      # Run only basic examples (no optional deps)
"""

import sys
import numpy as np

from histwords_analysis import HistWordsAnalyzer, list_available_datasets

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def print_section(title: str, level: int = 1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    else:
        print(f"\n--- {title} ---")


def run_basic_examples(eng: HistWordsAnalyzer):
    """Run examples that don't require optional dependencies."""

    # ============================================================
    # EXAMPLE 1: SEMANTIC CHANGE
    # ============================================================
    print_section("EXAMPLE 1: SEMANTIC CHANGE")

    # Classic example: "computer" used to mean "a person who computes"
    eng.semantic_change("computer", method="cosine")

    # "Gay" underwent significant semantic shift
    eng.semantic_change("gay", method="cosine")

    # Compare semantic change across words using neighbor analysis
    eng.semantic_change("broadcast", method="neighbors")

    # Compare multiple words (note: this re-analyzes words, so we use different ones)
    eng.compare_semantic_change(["telephone", "woman", "democracy", "science", "art"])

    # ============================================================
    # EXAMPLE 2: WORD SIMILARITY
    # ============================================================
    print_section("EXAMPLE 2: WORD SIMILARITY")

    # What was similar to "computer" in 1900 vs 1990?
    print_section("Computer's neighbors through time", level=2)
    eng.similar_words("computer", year=1900, top_k=10)
    eng.similar_words("computer", year=1950, top_k=10)
    eng.similar_words("computer", year=1990, top_k=10)

    # Track similarity between two specific words over time
    eng.similarity_over_time("gay", "happy")
    eng.similarity_over_time("gay", "homosexual")

    # Compare neighborhoods across time periods
    eng.neighborhood_comparison("woman", years=[1900, 1950, 1990], top_k=15)

    # ============================================================
    # EXAMPLE 3: WORD ANALOGIES
    # ============================================================
    print_section("EXAMPLE 3: WORD ANALOGIES")

    # Classic analogy: king is to man as queen is to woman
    eng.analogy("man", "king", "woman", year=1950)

    # Country-capital analogies
    eng.analogy("paris", "france", "london", year=1950)

    # How do analogies change over time?
    eng.analogy_over_time("man", "doctor", "woman", top_k=3)

    # Test gender analogies to see historical biases
    eng.gender_analogy_test(year=1900)
    eng.gender_analogy_test(year=1990)

    # ============================================================
    # EXAMPLE 4: WORKING WITH OTHER DATASETS
    # ============================================================
    print_section("EXAMPLE 4: OTHER DATASETS")

    # COHA (Corpus of Historical American English) - lemmatized
    try:
        coha = HistWordsAnalyzer("coha-lemma_sgns")
        coha.similar_words("woman", year=1900)
    except Exception as e:
        print(f"COHA not loaded: {e}")

    # Chinese dataset
    try:
        chi = HistWordsAnalyzer("chi-sim-all_sgns")
        print(f"\nChinese dataset years: {chi.years}")
        chi.vocabulary_stats(1980)
    except Exception as e:
        print(f"Chinese dataset: {e}")

    # ============================================================
    # EXAMPLE 5: CUSTOM ANALYSIS
    # ============================================================
    print_section("EXAMPLE 5: CUSTOM ANALYSIS PATTERNS")

    # Find words containing a pattern
    print("\nWords containing 'comput' in 1950:")
    matches = eng.find_words_by_pattern("comput", 1950)
    print(f"  Found {len(matches)} matches: {matches[:10]}")

    # Check if a word exists across time
    print("\nDoes 'television' exist across time?")
    existence = eng.word_exists("television")
    years_present = [y for y, exists in existence.items() if exists]
    years_absent = [y for y, exists in existence.items() if not exists]
    print(f"  Present in: {years_present}")
    if years_absent:
        print(f"  Absent in: {years_absent}")

    # Get raw vector for custom computations
    vec = eng.get_vector("democracy", 1950)
    if vec is not None:
        norm = np.linalg.norm(vec)
        print(f"\nVector for 'democracy' (1950): shape={vec.shape}, L2 norm={norm:.4f}")


def run_visualization_examples(eng: HistWordsAnalyzer):
    """Run visualization examples (requires matplotlib)."""

    if not HAS_MATPLOTLIB:
        print("\nSkipping visualization examples - matplotlib not installed")
        print("Install with: pip install matplotlib")
        return

    print_section("EXAMPLE 6: VISUALIZATION")

    # Plot semantic change over time for a word
    print_section("Plotting semantic change", level=2)
    eng.plot_semantic_change("computer")

    # Plot similarity between word pairs over time
    print_section("Plotting word pair similarities", level=2)
    eng.plot_similarity_over_time([
        ("gay", "happy"),
        ("gay", "homosexual"),
        ("woman", "work")
    ])

    # Visualize neighborhood evolution
    print_section("Plotting neighborhood evolution", level=2)
    eng.plot_neighborhood_evolution("woman", top_k=10)


def run_birth_death_examples(eng: HistWordsAnalyzer):
    """Run word birth/death detection examples."""

    print_section("EXAMPLE 7: WORD BIRTH/DEATH DETECTION")

    # Detect when new words entered the vocabulary
    print_section("Detecting word births", level=2)
    births = eng.detect_word_births(min_persistence=2)
    if births:
        total_births = sum(len(words) for words in births.values())
        print(f"\nTotal new words detected: {total_births}")

    # Detect when words left the vocabulary
    print_section("Detecting word deaths", level=2)
    deaths = eng.detect_word_deaths(min_absence=2)
    if deaths:
        total_deaths = sum(len(words) for words in deaths.values())
        print(f"\nTotal words that disappeared: {total_deaths}")

    # Analyze lifespan of specific words
    print_section("Word lifespan analysis", level=2)
    eng.word_lifespan("computer")
    eng.word_lifespan("typewriter")

    # Find neologisms from a specific era
    print_section("Finding neologisms (1950-1990)", level=2)
    neologisms = eng.find_neologisms(start_year=1950, end_year=1990, top_k=30)
    if neologisms:
        print(f"\nFirst neologism: '{neologisms[0][0]}' appeared in {neologisms[0][1]}")


def run_clustering_examples(eng: HistWordsAnalyzer):
    """Run semantic drift clustering examples (requires sklearn)."""

    if not HAS_SKLEARN:
        print("\nSkipping clustering examples - scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        return

    print_section("EXAMPLE 8: SEMANTIC DRIFT CLUSTERING")

    # Cluster words by their semantic drift patterns
    print_section("Clustering semantic drift patterns", level=2)
    clusters = eng.cluster_semantic_drift(n_clusters=5, sample_size=500)
    if clusters:
        print(f"\nCreated {clusters['n_clusters']} clusters")
        for cid, stats in clusters['cluster_stats'].items():
            print(f"  Cluster {cid}: {stats['size']} words, avg change: {stats['avg_change']:.4f}")

    # Find words that co-evolved with a target word
    print_section("Finding co-evolving words", level=2)
    co_evolving = eng.find_co_evolving_words("computer", top_k=15)
    if co_evolving:
        print(f"\nTop co-evolving word with 'computer': '{co_evolving[0][0]}' (similarity: {co_evolving[0][1]:.4f})")

    eng.find_co_evolving_words("woman", top_k=15)

    # Visualize drift clusters (requires matplotlib + sklearn)
    if HAS_MATPLOTLIB:
        print_section("Plotting drift clusters", level=2)
        eng.plot_drift_clusters(n_clusters=5, sample_size=300)


def run_cultural_shift_examples(eng: HistWordsAnalyzer):
    """Run cultural shift detection examples."""

    print_section("EXAMPLE 9: CULTURAL SHIFT DETECTION")

    # Automatically detect major cultural shifts
    print_section("Detecting cultural shifts", level=2)
    shifts = eng.detect_cultural_shifts(top_k=15, sample_size=1000)

    # Find the period with maximum average change
    if shifts:
        max_period = max(shifts.keys(), key=lambda p: np.mean([c for _, c in shifts[p]]))
        max_avg = np.mean([c for _, c in shifts[max_period]])
        print(f"\nPeriod with most change: {max_period[0]}-{max_period[1]} (avg: {max_avg:.4f})")
        print(f"Top changing words: {[w for w, _ in shifts[max_period][:5]]}")

    # Identify significant shift periods for specific words
    print_section("Identifying shift periods", level=2)
    for word in ["computer", "gay", "woman"]:
        result = eng.identify_shift_periods(word)
        if result and result.get('significant_shifts'):
            shifts_str = ", ".join([f"{s['from_year']}-{s['to_year']}" for s in result['significant_shifts']])
            print(f"  '{word}' significant shifts: {shifts_str}")

    # Analyze specific cultural eras
    print_section("Cultural era analysis: The Digital Age", level=2)
    eng.cultural_era_analysis(
        era_name="The Digital Age",
        start_year=1960,
        end_year=1990,
        keywords=["computer", "network", "data", "information", "software"]
    )

    print_section("Cultural era analysis: Women's Rights Movement", level=2)
    eng.cultural_era_analysis(
        era_name="Women's Rights Era",
        start_year=1920,
        end_year=1970,
        keywords=["woman", "vote", "equality", "rights", "work"]
    )

    # Visualize cultural shift intensity over time
    if HAS_MATPLOTLIB:
        print_section("Plotting cultural shift intensity", level=2)
        eng.plot_cultural_shifts(top_k=10, sample_size=500)


def print_summary():
    """Print summary of available features."""
    print_section("ALL EXAMPLES COMPLETE!")

    print("""
Available Features:

BASIC ANALYSIS:
  - semantic_change(word)           : Track meaning drift over time
  - compare_semantic_change(words)  : Compare change across words
  - similar_words(word, year)       : Find neighbors in a time period
  - similarity_over_time(w1, w2)    : Track pair similarity
  - analogy(a, b, c, year)          : Word analogies (a:b :: c:?)
  - gender_analogy_test(year)       : Test gender bias in analogies

VISUALIZATION (requires matplotlib):
  - plot_semantic_change(word)        : Plot semantic drift
  - plot_similarity_over_time(pairs)  : Plot pair similarities
  - plot_neighborhood_evolution(word) : Heatmap of neighbors
  - plot_drift_clusters()             : PCA of drift patterns
  - plot_cultural_shifts()            : Bar chart of shift intensity

WORD BIRTH/DEATH:
  - detect_word_births()            : Find vocabulary entries
  - detect_word_deaths()            : Find vocabulary exits
  - word_lifespan(word)             : Analyze word history
  - find_neologisms(start, end)     : Find new words in period

CLUSTERING (requires scikit-learn):
  - cluster_semantic_drift()        : Group words by drift
  - find_co_evolving_words(word)    : Find similar changers

CULTURAL ANALYSIS:
  - detect_cultural_shifts()        : Find top changers per decade
  - identify_shift_periods(word)    : Find significant shifts
  - cultural_era_analysis()         : Analyze historical eras
""")

    # Print dependency status
    print("Dependency Status:")
    print(f"  matplotlib:    {'INSTALLED' if HAS_MATPLOTLIB else 'NOT INSTALLED (pip install matplotlib)'}")
    print(f"  scikit-learn:  {'INSTALLED' if HAS_SKLEARN else 'NOT INSTALLED (pip install scikit-learn)'}")


def main():
    """Main entry point."""
    # Parse simple arguments
    quick_mode = "--quick" in sys.argv

    # List available datasets
    print("Available datasets:")
    for ds in list_available_datasets():
        print(f"  - {ds}")

    # Load English dataset (largest coverage: 1800-1990)
    print("\nLoading English dataset...")
    eng = HistWordsAnalyzer("eng-all_sgns")

    # Run basic examples (always)
    run_basic_examples(eng)

    print_section("DONE with basic examples!")

    if quick_mode:
        print("\nQuick mode: Skipping optional examples")
        print("Run without --quick to see all examples")
    else:
        # Run examples requiring optional dependencies
        run_visualization_examples(eng)
        run_birth_death_examples(eng)
        run_clustering_examples(eng)
        run_cultural_shift_examples(eng)

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
