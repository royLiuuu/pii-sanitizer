import unittest
from pii_sanitizer import PiiSanitizer, RunningMode


class TestWrapRephraseIntegration(unittest.TestCase):
    def setUp(self):
        # Use PRESIDIO mode for integration testing to verify real logic
        # independent of AWS Bedrock availability
        self.sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO)

    def test_wrap_rephrase_basic_integration(self):
        """
        Test that wrap_rephrase correctly anonymizes input before passing to the decorated function,
        and restores it afterwards using the real PiiSanitizer logic.
        """
        original_message = "My name is John Doe and my email is john.doe@example.com"

        @self.sanitizer.wrap_rephrase()
        def mock_llm_echo(anonymized_text: str) -> str:
            # Assert that PII has been replaced with placeholders inside the function
            self.assertNotIn("John Doe", anonymized_text)
            self.assertNotIn("john.doe@example.com", anonymized_text)
            self.assertIn("PERSON", anonymized_text)
            self.assertIn("EMAIL_ADDRESS", anonymized_text)
            return anonymized_text

        # The result should be the original message because the mock function just returns the anonymized text,
        # and wrap_rephrase restores the original values into the placeholders.
        restored_message = mock_llm_echo(original_message)

        self.assertEqual(restored_message, original_message)

    def test_wrap_rephrase_with_multiple_args_default(self):
        """
        Test that wrap_rephrase works when the decorated function has multiple arguments,
        defaulting to the first argument (index 0).
        """
        original_message = "Contact support at 555-0199."

        @self.sanitizer.wrap_rephrase()
        def mock_llm_response(content: str, timestamp: int) -> str:
            # Verify content is anonymized
            self.assertNotIn("555-0199", content)
            return f"Received at {timestamp}: {content}"

        result = mock_llm_response(original_message, 1234567890)

        # The placeholders in the LLM response should be restored to real values
        expected_result = "Received at 1234567890: Contact support at 555-0199."
        self.assertEqual(result, expected_result)

    def test_wrap_rephrase_arg_name(self):
        """
        Test targeting argument by name using arg_name parameter.
        """
        original_message = "My name is Alice."

        @self.sanitizer.wrap_rephrase(arg_name="prompt")
        def llm_call(model, prompt, temperature=0.7):
            # prompt should be anonymized
            if "Alice" in prompt:
                raise ValueError("PII leaked into function! 'Alice' found in prompt.")
            return f"Model {model} processed: {prompt}"

        # Call with keyword argument
        result = llm_call("gpt-4", prompt=original_message)
        self.assertEqual(result, "Model gpt-4 processed: My name is Alice.")

        # Call with positional argument (binding should still work)
        result_pos = llm_call("gpt-4", original_message)
        self.assertEqual(result_pos, "Model gpt-4 processed: My name is Alice.")

    def test_wrap_rephrase_arg_index(self):
        """
        Test targeting argument by index using arg_index parameter.
        """
        original_message = "Call me at 555-9999"

        # Target 2nd argument (index 1)
        @self.sanitizer.wrap_rephrase(arg_index=1)
        def process_data(id_val, data, meta=None):
            if "555-9999" in data:
                raise ValueError("PII leaked into function!")
            return f"{id_val}: {data}"

        result = process_data(101, original_message)
        self.assertEqual(result, "101: Call me at 555-9999")

    def test_wrap_rephrase_empty_string(self):
        @self.sanitizer.wrap_rephrase()
        def simple_func(text):
            return "Processed " + text

        self.assertEqual(simple_func(""), "Processed ")
        # When input is None, wrap_rephrase returns None immediately as per logic
        self.assertIsNone(simple_func(None))

    def test_wrap_rephrase_invalid_input_type(self):
        """
        Test that passing a non-string to the targeted argument raises an exception.
        """
        @self.sanitizer.wrap_rephrase()
        def math_op(x, y):
            return x + y
        
        # PiiSanitizer raises InvalidParamException for non-string input
        # We need to import it or check for Exception if not exposed easily in tests
        # Assuming InvalidParamException inherits from Exception
        with self.assertRaises(Exception) as context:
            math_op(1, 2)
        
        self.assertIn("Input message must be a string", str(context.exception))

if __name__ == '__main__':
    unittest.main()
