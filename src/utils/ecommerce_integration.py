# src/utils/ecommerce_integration.py
import requests
import os
import json
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EcommerceIntegration:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize e-commerce integration

        Args:
            api_key: API key for e-commerce platform
            base_url: Base URL for e-commerce API
        """
        self.api_key = api_key or os.environ.get("ECOMMERCE_API_KEY")
        self.base_url = base_url or os.environ.get("ECOMMERCE_API_URL", "https://api.ecommerce.example.com")

        # Map intents to actions
        self.intent_actions = {
            "purchase": self.purchase_item,
            "add_to_cart": self.add_to_cart,
            "search": self.search_items,
            "remove_from_cart": self.remove_from_cart,
            "checkout": self.checkout,
            "intent_to_buy": self.show_recommendations,
            "continue": self.continue_flow,
            "go_back": self.go_back,
            "show_items": self.show_items,
            "show_cart": self.show_cart,
            "confirm_order": self.confirm_order,
            "make_payment": self.make_payment,
            "ask_questions": self.ask_questions,
            "help": self.get_help,
            "cancel": self.cancel_action,
            "show_price_images": self.show_price_images,
            "change_quantity": self.change_quantity,
            "show_categories": self.show_categories,
            "show_description": self.show_description,
            "save_for_later": self.save_for_later
        }

    def execute_action(self, intent: str, user_id: str, confidence: float, **kwargs) -> Dict[str, Any]:
        """
        Execute action based on intent

        Args:
            intent: Recognized intent
            user_id: User ID
            confidence: Confidence score
            **kwargs: Additional parameters

        Returns:
            Action result
        """
        # Check confidence threshold
        if confidence < 0.7:
            return {
                "status": "low_confidence",
                "message": "Intent was recognized with low confidence. Please try again.",
                "confidence": confidence
            }

        # Check if intent exists
        if intent not in self.intent_actions:
            return {
                "status": "unknown_intent",
                "message": f"Intent '{intent}' is not supported"
            }

        # Execute action
        try:
            action_fn = self.intent_actions[intent]
            result = action_fn(user_id=user_id, **kwargs)

            # Add standard fields
            result.update({
                "intent": intent,
                "confidence": confidence
            })

            return result

        except Exception as e:
            logger.error(f"Error executing action for intent '{intent}': {e}")
            return {
                "status": "error",
                "message": f"Failed to execute action for intent '{intent}': {str(e)}",
                "intent": intent,
                "confidence": confidence
            }

    def _make_api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make request to e-commerce API"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            if method.lower() == 'get':
                response = requests.get(url, headers=headers, params=data)
            elif method.lower() == 'post':
                response = requests.post(url, headers=headers, json=data)
            elif method.lower() == 'put':
                response = requests.put(url, headers=headers, json=data)
            elif method.lower() == 'delete':
                response = requests.delete(url, headers=headers, params=data)
            else:
                return {"status": "error", "message": f"Unsupported method: {method}"}

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"API request error: {e}")
            return {"status": "error", "message": str(e)}

    # Action implementations
    def purchase_item(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Purchase an item directly"""
        if item_id:
            return self._make_api_request('post', '/orders/quick-purchase', {
                'user_id': user_id,
                'item_id': item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify which item you'd like to purchase",
                "required_params": ["item_id"]
            }

    def add_to_cart(self, user_id: str, item_id: Optional[str] = None, quantity: int = 1, **kwargs) -> Dict[str, Any]:
        """Add item to cart"""
        if item_id:
            return self._make_api_request('post', '/cart/items', {
                'user_id': user_id,
                'item_id': item_id,
                'quantity': quantity
            })
        else:
            # In a real implementation, you might try to extract the item from context or previous conversation
            return {
                "status": "additional_info_needed",
                "message": "Please specify which item you'd like to add to your cart",
                "required_params": ["item_id"]
            }
    def search_items(self, user_id: str, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
            """Search for items"""
            if query:
                return self._make_api_request('get', '/items/search', {
                    'user_id': user_id,
                    'query': query
                })
            else:
                return {
                    "status": "additional_info_needed",
                    "message": "What are you looking for?",
                    "required_params": ["query"]
                }

    def remove_from_cart(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Remove item from cart"""
        if item_id:
            return self._make_api_request('delete', '/cart/items', {
                'user_id': user_id,
                'item_id': item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Which item would you like to remove from your cart?",
                "required_params": ["item_id"]
            }

    def checkout(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Begin checkout process"""
        return self._make_api_request('post', '/checkout/start', {
            'user_id': user_id
        })

    def show_recommendations(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Show product recommendations"""
        return self._make_api_request('get', '/recommendations', {
            'user_id': user_id
        })

    def continue_flow(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Continue to next step in purchase flow"""
        return self._make_api_request('get', '/flow/continue', {
            'user_id': user_id
        })

    def go_back(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Go back to previous step"""
        return self._make_api_request('get', '/flow/back', {
            'user_id': user_id
        })

    def show_items(self, user_id: str, category: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show items, optionally filtered by category"""
        params = {'user_id': user_id}
        if category:
            params['category'] = category

        return self._make_api_request('get', '/items', params)

    def show_cart(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Show user's cart"""
        return self._make_api_request('get', '/cart', {
            'user_id': user_id
        })

    def confirm_order(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Confirm order"""
        return self._make_api_request('post', '/orders/confirm', {
            'user_id': user_id
        })

    def make_payment(self, user_id: str, payment_method: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Process payment"""
        params = {'user_id': user_id}
        if payment_method:
            params['payment_method'] = payment_method

        return self._make_api_request('post', '/payment/process', params)

    def ask_questions(self, user_id: str, question: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Ask questions about products or services"""
        if question:
            return self._make_api_request('post', '/support/question', {
                'user_id': user_id,
                'question': question
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "What's your question?",
                "required_params": ["question"]
            }

    def get_help(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Get customer support"""
        return self._make_api_request('get', '/support', {
            'user_id': user_id
        })

    def cancel_action(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Cancel current action or order"""
        return self._make_api_request('post', '/actions/cancel', {
            'user_id': user_id
        })

    def show_price_images(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show price and images for an item"""
        if item_id:
            return self._make_api_request('get', '/items/details', {
                'user_id': user_id,
                'item_id': item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Which item would you like to see?",
                "required_params": ["item_id"]
            }

    def change_quantity(self, user_id: str, item_id: Optional[str] = None, quantity: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Change quantity of an item in cart"""
        if item_id and quantity is not None:
            return self._make_api_request('put', '/cart/items', {
                'user_id': user_id,
                'item_id': item_id,
                'quantity': quantity
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and quantity",
                "required_params": ["item_id", "quantity"]
            }

    def show_categories(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Show available product categories"""
        return self._make_api_request('get', '/categories', {
            'user_id': user_id
        })

    def show_description(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show description of an item"""
        if item_id:
            return self._make_api_request('get', '/items/description', {
                'user_id': user_id,
                'item_id': item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Which item's description would you like to see?",
                "required_params": ["item_id"]
            }

    def save_for_later(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Save an item for later"""
        if item_id:
            return self._make_api_request('post', '/wishlist/items', {
                'user_id': user_id,
                'item_id': item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Which item would you like to save for later?",
                "required_params": ["item_id"]
            }
