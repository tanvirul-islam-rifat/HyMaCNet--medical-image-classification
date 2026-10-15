import copy
import unittest

from tools import check_results, metrics


class MetricTests(unittest.TestCase):
    def test_perfect_and_chance_kappa(self):
        self.assertAlmostEqual(metrics.confusion_metrics([[5, 0], [0, 5]])["kappa"], 1.0)
        self.assertAlmostEqual(metrics.confusion_metrics([[5, 5], [5, 5]])["kappa"], 0.0)

    def test_weighted_recall_is_accuracy(self):
        for data in metrics.benchmark_matrices().values():
            m = metrics.confusion_metrics(data["matrix"])
            self.assertAlmostEqual(m["weighted_recall"], m["accuracy"], places=12)

    def test_binary_example(self):
        m = metrics.confusion_metrics([[31, 11], [9, 105]])      # BreastMNIST
        self.assertEqual(m["n"], 156)
        self.assertAlmostEqual(m["accuracy"], 136 / 156)

    def test_labels_match_matrix_size(self):
        for name, data in metrics.benchmark_matrices().items():
            self.assertEqual(len(data["labels"]), len(data["matrix"]), name)
            self.assertTrue(all(len(row) == len(data["matrix"]) for row in data["matrix"]), name)


class WilcoxonTests(unittest.TestCase):
    def test_all_positive_five_pairs(self):
        r = metrics.wilcoxon_signed_rank_exact([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
        self.assertAlmostEqual(r["p_one_sided"], 1 / 32)
        self.assertAlmostEqual(r["p_two_sided"], 1 / 16)
        self.assertEqual(r["rank_biserial"], 1.0)

    def test_mixed_signs_not_significant(self):
        r = metrics.wilcoxon_signed_rank_exact([1, 2, 3, 4, 5], [2, 1, 4, 3, 6])
        self.assertGreater(r["p_one_sided"], 0.05)

    def test_rejects_zero_differences(self):
        with self.assertRaises(ValueError):
            metrics.wilcoxon_signed_rank_exact([1, 2], [1, 3])


class CheckerTests(unittest.TestCase):
    """The checker passes on the released data and rejects corrupted copies of it."""

    def setUp(self):
        self.base = check_results.per_dataset("base")
        self.small = check_results.per_dataset("small")

    def failures(self, **kwargs):
        c = check_results.run(check_results.Checks(quiet=True), **kwargs)
        return [name for ok, name, _ in c.rows if not ok]

    def test_released_results_pass(self):
        self.assertEqual(self.failures(), [])

    def test_rejects_accuracy_that_contradicts_recall(self):
        base = copy.deepcopy(self.base)
        base[2]["oa"] = "97.83"                                  # OrganAMNIST
        self.assertTrue(any("weighted recall" in f for f in self.failures(base=base)))

    def test_rejects_accuracy_that_contradicts_confusion_matrix(self):
        base = copy.deepcopy(self.base)
        row = next(r for r in base if r["dataset"] == "OrganSMNIST")
        row["oa"], row["recall"] = "85.74", "0.8574"
        self.assertTrue(any("confusion matrices" in f for f in self.failures(base=base)))

    def test_rejects_impossible_f1(self):
        small = copy.deepcopy(self.small)
        small[0]["f1"] = "0.9990"
        self.assertTrue(any("attainable" in f for f in self.failures(small=small)))

    def test_rejects_wrong_seed_statistics(self):
        stats = copy.deepcopy(metrics.seed_summary())
        stats["oa"]["hymac_mean"] = 0.9306                      # what a mistyped seed run gives
        self.assertTrue(any("Table 5.7" in f for f in self.failures(seeds_override=stats)))


if __name__ == "__main__":
    unittest.main()
