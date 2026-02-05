"""
Retrieval Evaluation Framework

Build a system to evaluate and optimize retrieval quality using test questions and ground truth.
This framework provides comprehensive evaluation capabilities including precision@k, recall@k,
MRR metrics, strategy comparison, and actionable insights generation.
"""

import math
import random
from typing import Dict, List, Optional, Any, Callable, Set, Tuple


class TestDatasetManager:
    """
    Manages test datasets with questions and ground truth mappings.

    Handles creation, validation, and access to test data used for
    evaluating retrieval systems.
    """

    def __init__(self):
        """Initialize the TestDatasetManager."""
        self._questions: List[str] = []
        self._ground_truth: Dict[str, List[str]] = {}
        self._corpus: List[Dict[str, Any]] = []

    def create_test_dataset(
        self,
        questions: List[str],
        ground_truth: Dict[str, List[str]],
        corpus: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a test dataset with questions and expected relevant chunks.

        Args:
            questions: List of test questions
            ground_truth: Mapping of questions to relevant chunk IDs
            corpus: List of document chunks with 'id' and 'text' keys

        Returns:
            True if dataset was created successfully
        """
        if not questions or not ground_truth or not corpus:
            return False

        self._questions = questions
        self._ground_truth = ground_truth
        self._corpus = corpus

        return True

    def generate_synthetic_dataset(self, size: int = 10) -> Dict[str, Any]:
        """
        Generate a synthetic test dataset for demonstration.

        Args:
            size: Number of test questions to generate

        Returns:
            Dictionary containing questions, ground_truth, and corpus
        """
        # Sample document corpus covering various RAG-related topics
        corpus = [
            {"id": "chunk_ml_intro", "text": "Machine learning algorithms learn patterns from data to make predictions."},
            {"id": "chunk_ml_training", "text": "Training machine learning models requires large datasets and computational resources."},
            {"id": "chunk_nn_basics", "text": "Neural networks consist of layers of interconnected nodes that process information."},
            {"id": "chunk_nn_deep", "text": "Deep neural networks have multiple hidden layers for complex pattern recognition."},
            {"id": "chunk_rag_intro", "text": "RAG systems combine retrieval with generation for more accurate responses."},
            {"id": "chunk_rag_benefits", "text": "Retrieval-augmented generation reduces hallucinations by grounding responses in facts."},
            {"id": "chunk_embeddings", "text": "Embeddings convert text into dense vectors that capture semantic meaning."},
            {"id": "chunk_vectors", "text": "Vector databases store embeddings for efficient similarity search operations."},
            {"id": "chunk_nlp_intro", "text": "Natural language processing enables computers to understand human language."},
            {"id": "chunk_nlp_tasks", "text": "Common NLP tasks include sentiment analysis, translation, and summarization."},
            {"id": "chunk_python", "text": "Python is widely used for machine learning and data science applications."},
            {"id": "chunk_transformers", "text": "Transformer models use attention mechanisms for processing sequential data."},
        ]

        # Test questions with ground truth mappings
        test_items = [
            ("What is machine learning?", ["chunk_ml_intro", "chunk_ml_training"]),
            ("How do neural networks work?", ["chunk_nn_basics", "chunk_nn_deep"]),
            ("Explain RAG systems", ["chunk_rag_intro", "chunk_rag_benefits"]),
            ("What are embeddings?", ["chunk_embeddings", "chunk_vectors"]),
            ("What is NLP?", ["chunk_nlp_intro", "chunk_nlp_tasks"]),
            ("How do transformers work?", ["chunk_transformers"]),
            ("What programming language for ML?", ["chunk_python"]),
            ("How to train AI models?", ["chunk_ml_training", "chunk_nn_deep"]),
            ("What is retrieval augmented generation?", ["chunk_rag_intro", "chunk_rag_benefits"]),
            ("How do vector databases work?", ["chunk_vectors", "chunk_embeddings"]),
        ]

        # Select requested number of items
        selected = test_items[:min(size, len(test_items))]

        questions = [q for q, _ in selected]
        ground_truth = {q: chunks for q, chunks in selected}

        self.create_test_dataset(questions, ground_truth, corpus)

        return {
            "questions": questions,
            "ground_truth": ground_truth,
            "corpus": corpus
        }

    def get_questions(self) -> List[str]:
        """Return the list of test questions."""
        return self._questions

    def get_ground_truth(self, question: str) -> List[str]:
        """Get the relevant chunk IDs for a question."""
        return self._ground_truth.get(question, [])

    def get_corpus(self) -> List[Dict[str, Any]]:
        """Return the document corpus."""
        return self._corpus

    def get_corpus_ids(self) -> Set[str]:
        """Return all chunk IDs in the corpus."""
        return {chunk["id"] for chunk in self._corpus}


class MetricsCalculator:
    """
    Calculates retrieval evaluation metrics.

    Implements precision@k, recall@k, and Mean Reciprocal Rank (MRR)
    for evaluating retrieval quality.
    """

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate precision@k for a single query.

        Precision@k = (relevant items in top k) / k

        Args:
            retrieved: List of retrieved chunk IDs (ordered by rank)
            relevant: List of relevant chunk IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Precision score between 0 and 1
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_in_top_k = sum(1 for item in top_k if item in relevant_set)

        return relevant_in_top_k / k

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate recall@k for a single query.

        Recall@k = (relevant items in top k) / (total relevant items)

        Args:
            retrieved: List of retrieved chunk IDs (ordered by rank)
            relevant: List of relevant chunk IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Recall score between 0 and 1
        """
        if not relevant or k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_in_top_k = sum(1 for item in top_k if item in relevant_set)

        return relevant_in_top_k / len(relevant)

    def reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate reciprocal rank for a single query.

        RR = 1 / (rank of first relevant item)

        Args:
            retrieved: List of retrieved chunk IDs (ordered by rank)
            relevant: List of relevant chunk IDs (ground truth)

        Returns:
            Reciprocal rank between 0 and 1
        """
        relevant_set = set(relevant)

        for rank, item in enumerate(retrieved, start=1):
            if item in relevant_set:
                return 1.0 / rank

        return 0.0

    def mean_reciprocal_rank(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across all queries.

        MRR = (1/n) * sum(RR for each query)

        Args:
            all_retrieved: List of retrieved results for each query
            all_relevant: List of relevant items for each query

        Returns:
            MRR score between 0 and 1
        """
        if not all_retrieved or len(all_retrieved) != len(all_relevant):
            return 0.0

        total_rr = sum(
            self.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in zip(all_retrieved, all_relevant)
        )

        return total_rr / len(all_retrieved)

    def calculate_all_metrics(
        self,
        retrieved: List[str],
        relevant: List[str],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Calculate all metrics for a single query.

        Args:
            retrieved: List of retrieved chunk IDs
            relevant: List of relevant chunk IDs
            k_values: List of k values for precision@k and recall@k

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved, relevant, k)
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved, relevant, k)

        metrics["reciprocal_rank"] = self.reciprocal_rank(retrieved, relevant)

        return metrics


class MockRetrievalStrategy:
    """
    Mock retrieval strategy for testing and demonstration.

    Simulates different retrieval behaviors without requiring
    actual embedding models or vector databases.
    """

    def __init__(
        self,
        name: str,
        accuracy: float = 0.7,
        noise_level: float = 0.2
    ):
        """
        Initialize a mock retrieval strategy.

        Args:
            name: Name of the strategy
            accuracy: Probability of including relevant items (0-1)
            noise_level: Probability of including irrelevant items (0-1)
        """
        self.name = name
        self.accuracy = accuracy
        self.noise_level = noise_level
        self._seed = hash(name) % (2**32)

    def retrieve(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
        relevant_ids: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Simulate retrieval for a query.

        Args:
            query: Query string (used for seed)
            corpus: Document corpus
            relevant_ids: Ground truth relevant IDs
            top_k: Number of results to return

        Returns:
            List of retrieved chunk IDs
        """
        # Use deterministic random for reproducibility
        rng = random.Random(self._seed + hash(query))

        all_ids = [chunk["id"] for chunk in corpus]
        irrelevant_ids = [cid for cid in all_ids if cid not in relevant_ids]

        retrieved = []

        # Add relevant items based on accuracy
        for rel_id in relevant_ids:
            if rng.random() < self.accuracy and len(retrieved) < top_k:
                retrieved.append(rel_id)

        # Add some noise (irrelevant items)
        for irr_id in irrelevant_ids:
            if rng.random() < self.noise_level and len(retrieved) < top_k:
                retrieved.append(irr_id)

        # Fill remaining slots with random items
        remaining = [cid for cid in all_ids if cid not in retrieved]
        rng.shuffle(remaining)
        while len(retrieved) < top_k and remaining:
            retrieved.append(remaining.pop())

        # Shuffle to simulate imperfect ranking
        rng.shuffle(retrieved)

        return retrieved[:top_k]


class StrategyComparator:
    """
    Compares multiple retrieval strategies on the same test dataset.
    """

    def __init__(self, metrics_calculator: MetricsCalculator):
        """Initialize with a metrics calculator."""
        self._metrics = metrics_calculator
        self._strategies: List[MockRetrievalStrategy] = []
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_strategy(self, strategy: MockRetrievalStrategy) -> None:
        """Add a retrieval strategy for comparison."""
        self._strategies.append(strategy)

    def create_default_strategies(self) -> List[MockRetrievalStrategy]:
        """
        Create a set of default strategies with different characteristics.

        Returns:
            List of mock strategies for comparison
        """
        strategies = [
            MockRetrievalStrategy("high_precision", accuracy=0.9, noise_level=0.1),
            MockRetrievalStrategy("high_recall", accuracy=0.95, noise_level=0.4),
            MockRetrievalStrategy("balanced", accuracy=0.75, noise_level=0.25),
            MockRetrievalStrategy("baseline", accuracy=0.5, noise_level=0.5),
        ]

        for s in strategies:
            self.add_strategy(s)

        return strategies

    def evaluate_strategy(
        self,
        strategy: MockRetrievalStrategy,
        dataset_manager: TestDatasetManager,
        k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate a single strategy on the test dataset.

        Args:
            strategy: Retrieval strategy to evaluate
            dataset_manager: Test dataset manager
            k_values: List of k values for metrics

        Returns:
            Dictionary of evaluation results
        """
        questions = dataset_manager.get_questions()
        corpus = dataset_manager.get_corpus()

        all_retrieved: List[List[str]] = []
        all_relevant: List[List[str]] = []
        per_query_metrics: List[Dict[str, Any]] = []

        for question in questions:
            relevant = dataset_manager.get_ground_truth(question)
            retrieved = strategy.retrieve(question, corpus, relevant, top_k=max(k_values))

            all_retrieved.append(retrieved)
            all_relevant.append(relevant)

            query_metrics = self._metrics.calculate_all_metrics(retrieved, relevant, k_values)
            per_query_metrics.append({
                "question": question,
                "retrieved": retrieved,
                "relevant": relevant,
                "metrics": query_metrics
            })

        # Aggregate metrics
        aggregated = {}
        for k in k_values:
            precision_key = f"precision@{k}"
            recall_key = f"recall@{k}"

            precisions = [q["metrics"][precision_key] for q in per_query_metrics]
            recalls = [q["metrics"][recall_key] for q in per_query_metrics]

            aggregated[precision_key] = sum(precisions) / len(precisions) if precisions else 0
            aggregated[recall_key] = sum(recalls) / len(recalls) if recalls else 0

        aggregated["mrr"] = self._metrics.mean_reciprocal_rank(all_retrieved, all_relevant)

        return {
            "strategy_name": strategy.name,
            "aggregated_metrics": aggregated,
            "per_query_results": per_query_metrics,
            "num_queries": len(questions)
        }

    def compare_all(
        self,
        dataset_manager: TestDatasetManager,
        k_values: List[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all registered strategies.

        Args:
            dataset_manager: Test dataset manager
            k_values: List of k values (default: [3, 5, 10])

        Returns:
            Dictionary mapping strategy names to their results
        """
        k_values = k_values or [3, 5, 10]

        self._results = {}
        for strategy in self._strategies:
            self._results[strategy.name] = self.evaluate_strategy(
                strategy, dataset_manager, k_values
            )

        return self._results

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Return the comparison results."""
        return self._results


class ReportGenerator:
    """
    Generates evaluation reports with actionable insights.
    """

    def generate_report(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Args:
            comparison_results: Results from strategy comparison
            k_values: K values used in evaluation

        Returns:
            Report dictionary with metrics, rankings, and insights
        """
        if not comparison_results:
            return {"error": "No comparison results available"}

        # Extract key metrics for ranking
        strategy_summaries = []
        for name, results in comparison_results.items():
            metrics = results["aggregated_metrics"]
            strategy_summaries.append({
                "name": name,
                "precision@5": metrics.get("precision@5", 0),
                "recall@5": metrics.get("recall@5", 0),
                "mrr": metrics.get("mrr", 0),
                "f1@5": self._calculate_f1(
                    metrics.get("precision@5", 0),
                    metrics.get("recall@5", 0)
                )
            })

        # Rank strategies by different metrics
        rankings = {
            "by_precision": sorted(strategy_summaries, key=lambda x: x["precision@5"], reverse=True),
            "by_recall": sorted(strategy_summaries, key=lambda x: x["recall@5"], reverse=True),
            "by_mrr": sorted(strategy_summaries, key=lambda x: x["mrr"], reverse=True),
            "by_f1": sorted(strategy_summaries, key=lambda x: x["f1@5"], reverse=True)
        }

        # Generate insights
        insights = self._generate_insights(strategy_summaries, rankings)

        # Generate recommendations
        recommendations = self._generate_recommendations(strategy_summaries, rankings)

        return {
            "summary": {
                "num_strategies": len(comparison_results),
                "k_values_evaluated": k_values,
                "total_queries": next(iter(comparison_results.values()))["num_queries"]
            },
            "strategy_summaries": strategy_summaries,
            "rankings": rankings,
            "insights": insights,
            "recommendations": recommendations,
            "detailed_results": comparison_results
        }

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _generate_insights(
        self,
        summaries: List[Dict[str, Any]],
        rankings: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate analytical insights from the results."""
        insights = []

        # Best overall performer
        best_f1 = rankings["by_f1"][0]
        insights.append(
            f"Best overall performer: '{best_f1['name']}' with F1@5 score of {best_f1['f1@5']:.3f}"
        )

        # Precision vs Recall trade-off
        best_precision = rankings["by_precision"][0]
        best_recall = rankings["by_recall"][0]

        if best_precision["name"] != best_recall["name"]:
            insights.append(
                f"Trade-off observed: '{best_precision['name']}' leads in precision ({best_precision['precision@5']:.3f}) "
                f"while '{best_recall['name']}' leads in recall ({best_recall['recall@5']:.3f})"
            )
        else:
            insights.append(
                f"'{best_precision['name']}' leads in both precision and recall"
            )

        # MRR analysis
        best_mrr = rankings["by_mrr"][0]
        insights.append(
            f"Best ranking quality: '{best_mrr['name']}' with MRR of {best_mrr['mrr']:.3f}"
        )

        # Performance spread
        precisions = [s["precision@5"] for s in summaries]
        if max(precisions) - min(precisions) > 0.3:
            insights.append(
                "Large performance spread observed between strategies - "
                "strategy selection significantly impacts results"
            )

        return insights

    def _generate_recommendations(
        self,
        summaries: List[Dict[str, Any]],
        rankings: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        best_f1 = rankings["by_f1"][0]
        best_precision = rankings["by_precision"][0]
        best_recall = rankings["by_recall"][0]

        # General recommendation
        recommendations.append(
            f"For balanced performance, use '{best_f1['name']}' strategy"
        )

        # Use case specific recommendations
        if best_precision["name"] != best_f1["name"]:
            recommendations.append(
                f"For applications requiring high accuracy (e.g., legal, medical), "
                f"consider '{best_precision['name']}' for its superior precision"
            )

        if best_recall["name"] != best_f1["name"]:
            recommendations.append(
                f"For comprehensive search (e.g., research, discovery), "
                f"consider '{best_recall['name']}' for its superior recall"
            )

        # Improvement suggestions
        worst_f1 = rankings["by_f1"][-1]
        if worst_f1["f1@5"] < 0.5:
            recommendations.append(
                f"'{worst_f1['name']}' shows poor performance (F1={worst_f1['f1@5']:.3f}) - "
                "consider tuning or replacing this strategy"
            )

        return recommendations

    def format_report_text(self, report: Dict[str, Any]) -> str:
        """
        Format the report as human-readable text.

        Args:
            report: Report dictionary

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("RETRIEVAL EVALUATION REPORT")
        lines.append("=" * 60)

        # Summary
        summary = report["summary"]
        lines.append(f"\nStrategies evaluated: {summary['num_strategies']}")
        lines.append(f"K values: {summary['k_values_evaluated']}")
        lines.append(f"Total test queries: {summary['total_queries']}")

        # Strategy summaries
        lines.append("\n" + "-" * 40)
        lines.append("STRATEGY PERFORMANCE SUMMARY")
        lines.append("-" * 40)

        for s in report["strategy_summaries"]:
            lines.append(f"\n{s['name']}:")
            lines.append(f"  Precision@5: {s['precision@5']:.3f}")
            lines.append(f"  Recall@5:    {s['recall@5']:.3f}")
            lines.append(f"  F1@5:        {s['f1@5']:.3f}")
            lines.append(f"  MRR:         {s['mrr']:.3f}")

        # Insights
        lines.append("\n" + "-" * 40)
        lines.append("INSIGHTS")
        lines.append("-" * 40)

        for insight in report["insights"]:
            lines.append(f"• {insight}")

        # Recommendations
        lines.append("\n" + "-" * 40)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)

        for rec in report["recommendations"]:
            lines.append(f"→ {rec}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


class RetrievalEvaluationFramework:
    """
    Main orchestrator for retrieval evaluation.

    Coordinates test dataset management, strategy comparison,
    metrics calculation, and report generation.
    """

    def __init__(self):
        """Initialize the RetrievalEvaluationFramework instance."""
        self._dataset_manager = TestDatasetManager()
        self._metrics_calculator = MetricsCalculator()
        self._strategy_comparator = StrategyComparator(self._metrics_calculator)
        self._report_generator = ReportGenerator()
        self._k_values = [3, 5, 10]

    def create_test_dataset(
        self,
        questions: List[str],
        ground_truth: Dict[str, List[str]],
        corpus: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a custom test dataset.

        Args:
            questions: List of test questions
            ground_truth: Mapping of questions to relevant chunk IDs
            corpus: Document corpus

        Returns:
            True if successful
        """
        return self._dataset_manager.create_test_dataset(questions, ground_truth, corpus)

    def generate_synthetic_dataset(self, size: int = 10) -> Dict[str, Any]:
        """
        Generate a synthetic test dataset.

        Args:
            size: Number of test questions

        Returns:
            Generated dataset
        """
        return self._dataset_manager.generate_synthetic_dataset(size)

    def get_dataset_manager(self) -> TestDatasetManager:
        """Return the dataset manager."""
        return self._dataset_manager

    def get_metrics_calculator(self) -> MetricsCalculator:
        """Return the metrics calculator."""
        return self._metrics_calculator

    def add_strategy(self, strategy: MockRetrievalStrategy) -> None:
        """Add a retrieval strategy for evaluation."""
        self._strategy_comparator.add_strategy(strategy)

    def create_default_strategies(self) -> List[MockRetrievalStrategy]:
        """Create default set of strategies for comparison."""
        return self._strategy_comparator.create_default_strategies()

    def set_k_values(self, k_values: List[int]) -> None:
        """Set k values for evaluation metrics."""
        self._k_values = k_values

    def evaluate_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all registered strategies.

        Returns:
            Comparison results
        """
        return self._strategy_comparator.compare_all(
            self._dataset_manager,
            self._k_values
        )

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate evaluation report.

        Returns:
            Report dictionary with insights and recommendations
        """
        results = self._strategy_comparator.get_results()
        return self._report_generator.generate_report(results, self._k_values)

    def get_formatted_report(self) -> str:
        """
        Get human-readable report text.

        Returns:
            Formatted report string
        """
        report = self.generate_report()
        return self._report_generator.format_report_text(report)

    def execute(self) -> Dict[str, Any]:
        """
        Main entry point demonstrating complete functionality.

        Returns:
            Dictionary with evaluation results and report
        """
        # Step 1: Create test dataset
        dataset = self.generate_synthetic_dataset(size=8)

        # Step 2: Create strategies for comparison
        strategies = self.create_default_strategies()

        # Step 3: Evaluate all strategies
        evaluation_results = self.evaluate_strategies()

        # Step 4: Generate report
        report = self.generate_report()

        # Step 5: Get formatted report
        formatted_report = self.get_formatted_report()

        return {
            "dataset": {
                "questions": dataset["questions"],
                "corpus_size": len(dataset["corpus"])
            },
            "strategies_evaluated": [s.name for s in strategies],
            "evaluation_results": evaluation_results,
            "report": report,
            "formatted_report": formatted_report
        }


def create_retrieval_evaluation_framework() -> RetrievalEvaluationFramework:
    """
    Factory function for creating RetrievalEvaluationFramework instances.

    Returns:
        RetrievalEvaluationFramework: A new instance of RetrievalEvaluationFramework
    """
    return RetrievalEvaluationFramework()
