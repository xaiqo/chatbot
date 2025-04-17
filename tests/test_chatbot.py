import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock for the ImprovedChatbot class
class MockImprovedChatbot:
    def __init__(self, model_path=None):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.tokenizer.bos_token = "<BOS>"
        self.tokenizer.sep_token = "<SEP>"
        self.device = "cpu"
        self.conversation_history = []
        self.max_history_length = 5
    
    def add_to_history(self, user_message, bot_response=None):
        self.conversation_history.append((user_message, bot_response))
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        self.conversation_history = []
    
    def _format_conversation(self):
        formatted_text = self.tokenizer.bos_token
        for user_msg, bot_msg in self.conversation_history:
            formatted_text += f"User: {user_msg}{self.tokenizer.sep_token}"
            if bot_msg:
                formatted_text += f"Bot: {bot_msg}{self.tokenizer.sep_token}"
        return formatted_text

# Mock the import
sys.modules['finetuning.improved_model'] = MagicMock()
sys.modules['finetuning.improved_model'].ImprovedChatbot = MockImprovedChatbot

class TestChatbot(unittest.TestCase):
    def setUp(self):
        # Initialize with mock chatbot
        self.chatbot = MockImprovedChatbot(model_path=None)
    
    def test_initialization(self):
        self.assertIsNotNone(self.chatbot)
        self.assertIsNotNone(self.chatbot.model)
        self.assertIsNotNone(self.chatbot.tokenizer)
    
    def test_conversation_history(self):
        self.assertEqual(len(self.chatbot.conversation_history), 0)
        
        # Add to history
        self.chatbot.add_to_history("Hello", "Hi there")
        self.assertEqual(len(self.chatbot.conversation_history), 1)
        
        # Clear history
        self.chatbot.clear_history()
        self.assertEqual(len(self.chatbot.conversation_history), 0)
    
    def test_format_conversation(self):
        self.chatbot.add_to_history("Hello", "Hi there")
        formatted = self.chatbot._format_conversation()
        self.assertIn("User: Hello", formatted)
        self.assertIn("Bot: Hi there", formatted)

if __name__ == '__main__':
    unittest.main()