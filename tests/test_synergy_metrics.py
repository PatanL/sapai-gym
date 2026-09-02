import unittest

from sapai import Pet, Team
from sapai_gym1.rl.synergy_metrics import detect_team_synergies


class TestSynergyMetrics(unittest.TestCase):
    def detect(self, *pets):
        return detect_team_synergies(Team(list(pets)))

    def test_elephant_hurt_lines_require_order(self):
        ordered = self.detect(Pet("elephant"), Pet("camel"), Pet("wolverine"))
        self.assertIn("elephant_camel_components", ordered)
        self.assertIn("elephant_camel_ordered", ordered)
        self.assertIn("elephant_hurt_wolverine", ordered)

        wrong_order = self.detect(Pet("camel"), Pet("elephant"), Pet("wolverine"))
        self.assertIn("elephant_camel_components", wrong_order)
        self.assertNotIn("elephant_camel_ordered", wrong_order)
        self.assertNotIn("elephant_hurt_wolverine", wrong_order)

    def test_shark_fly_and_faint_core(self):
        found = self.detect(Pet("deer"), Pet("turtle"), Pet("shark"), Pet("fly"))
        self.assertIn("shark_fly", found)
        self.assertIn("shark_faint_core", found)

    def test_deer_must_be_immediately_ahead_of_parrot(self):
        self.assertIn(
            "deer_parrot_ordered",
            self.detect(Pet("deer"), Pet("parrot")),
        )
        separated = self.detect(Pet("deer"), Pet("fish"), Pet("parrot"))
        self.assertIn("deer_parrot_components", separated)
        self.assertNotIn("deer_parrot_ordered", separated)

    def test_mammoth_requires_mushroom_and_tiger_behind(self):
        mammoth = Pet("mammoth")
        mammoth.status = "status-extra-life"
        found = self.detect(mammoth, Pet("tiger"))
        self.assertIn("mammoth_mushroom_tiger_components", found)
        self.assertIn("mammoth_mushroom_tiger_ordered", found)

        reversed_mammoth = Pet("mammoth")
        reversed_mammoth.status = "status-extra-life"
        reversed_found = self.detect(Pet("tiger"), reversed_mammoth)
        self.assertIn("mammoth_mushroom_tiger_components", reversed_found)
        self.assertNotIn("mammoth_mushroom_tiger_ordered", reversed_found)


if __name__ == "__main__":
    unittest.main()
