import unittest
from pydantic import ValidationError
from lite.models import validate_factors
from test_workflow import FACTORS

class FactorMetadataTests(unittest.TestCase):
    def test_revision_metadata_retained(self):
        factors={name:{**f,'concise_observation':'observed','concise_justification':'reason','concise_specification':'test plan'} for name,f in FACTORS.items()}
        self.assertEqual(validate_factors(factors),factors)

    def test_metadata_does_not_replace_required_expression(self):
        factors={name:{**f,'concise_specification':'not an expression'} for name,f in FACTORS.items()}
        del factors[next(iter(factors))]['expression']
        with self.assertRaises(ValidationError):validate_factors(factors)

    def test_unknown_field_still_rejected(self):
        factors={name:{**f,'unexpected':'value'} for name,f in FACTORS.items()}
        with self.assertRaises(ValidationError):validate_factors(factors)
