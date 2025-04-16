import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_model import ImprovedChatbot

class TestChatbot(unittest.TestCase):
    def setUp(self):
        # Initialize with no model to use base GPT-2
        self.chatbot = ImprovedChatbot(model_path=None)
    
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