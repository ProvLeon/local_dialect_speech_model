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
        # Expanded to align with lean prompts canonical intents (prompts_lean.csv)
        self.intent_actions = {
            # Core / navigation
            "go_home": self.go_home,
            "continue": self.continue_flow,
            "go_back": self.go_back,
            "show_cart": self.show_cart,
            "open_wishlist": self.view_wishlist,          # alias
            "view_wishlist": self.view_wishlist,
            "open_account": self.open_account,
            "open_orders": self.show_orders,
            "show_orders": self.show_orders,
            "help": self.get_help,
            "start_live_chat": self.start_live_chat,
            "show_faqs": self.show_faqs,

            # Discovery / search & filters
            "search": self.search_items,
            "apply_filter": self.apply_filter,
            "clear_filter": self.clear_filter,
            "sort_items": self.sort_items,
            "show_similar_items": self.show_similar_items,
            "show_categories": self.show_categories,

            # Product info
            "show_description": self.show_description,
            "show_price": self.show_price,                # new alias vs show_price_images
            "show_reviews": self.show_reviews,
            "show_price_images": self.show_price_images,  # legacy compatibility

            # Cart & purchase
            "add_to_cart": self.add_to_cart,
            "remove_from_cart": self.remove_from_cart,
            "change_quantity": self.change_quantity,
            "set_quantity": self.set_quantity,
            "clear_cart": self.clear_cart,
            "save_for_later": self.save_for_later,
            "checkout": self.checkout,
            "confirm_order": self.confirm_order,
            "make_payment": self.make_payment,
            "cancel_order": self.cancel_order,
            "purchase": self.purchase_item,
            "intent_to_buy": self.show_recommendations,

            # Variant / attributes
            "select_color": self.select_color,
            "select_size": self.select_size,
            "change_color": self.change_color,
            "change_size": self.change_size,

            # Addresses
            "show_addresses": self.show_addresses,
            "add_address": self.add_address,
            "remove_address": self.remove_address,
            "set_default_address": self.set_default_address,
            "manage_address": self.manage_address,  # legacy aggregate

            # Orders & post-purchase
            "track_order": self.track_order,
            "show_order_status": self.show_order_status,
            "order_not_arrived": self.order_not_arrived,
            "return_item": self.return_item,
            "exchange_item": self.exchange_item,
            "refund_status": self.refund_status,
            "change_order": self.change_order,

            # Coupons / promotions
            "apply_coupon": self.apply_coupon,
            "remove_coupon": self.remove_coupon,

            # Notifications (granular)
            "enable_order_updates": self.enable_order_updates,
            "disable_order_updates": self.disable_order_updates,
            "enable_price_alert": self.enable_price_alert,
            "disable_price_alert": self.disable_price_alert,
            "manage_notifications": self.manage_notifications,  # legacy aggregate

            # Profile / account management
            "manage_profile": self.manage_profile,
            "open_account_settings": self.manage_profile,  # alias

            # Support / misc
            "ask_questions": self.ask_questions,
            "cancel": self.cancel_action,  # legacy generic cancel
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

    # New intent handlers
    def select_color(self, user_id: str, color: Optional[str] = None, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Select a color for an item"""
        if color and item_id:
            return self._make_api_request('put', '/items/select-color', {
                'user_id': user_id,
                'item_id': item_id,
                'color': color
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and color",
                "required_params": ["item_id", "color"]
            }

    def select_size(self, user_id: str, size: Optional[str] = None, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Select a size for an item"""
        if size and item_id:
            return self._make_api_request('put', '/items/select-size', {
                'user_id': user_id,
                'item_id': item_id,
                'size': size
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and size",
                "required_params": ["item_id", "size"]
            }

    def change_color(self, user_id: str, new_color: Optional[str] = None, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Change the color of an item"""
        if new_color and item_id:
            return self._make_api_request('put', '/items/change-color', {
                'user_id': user_id,
                'item_id': item_id,
                'new_color': new_color
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and new color",
                "required_params": ["item_id", "new_color"]
            }

    def change_size(self, user_id: str, new_size: Optional[str] = None, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Change the size of an item"""
        if new_size and item_id:
            return self._make_api_request('put', '/items/change-size', {
                'user_id': user_id,
                'item_id': item_id,
                'new_size': new_size
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and new size",
                "required_params": ["item_id", "new_size"]
            }

    def change_order(self, user_id: str, order_id: Optional[str] = None, changes: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Change or modify an order"""
        if order_id:
            return self._make_api_request('put', '/orders/modify', {
                'user_id': user_id,
                'order_id': order_id,
                'changes': changes or {}
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify which order to change",
                "required_params": ["order_id"]
            }

    def manage_address(self, user_id: str, action: str = "view", address_data: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Manage user addresses"""
        if action == "view":
            return self._make_api_request('get', '/addresses', {
                'user_id': user_id
            })
        elif action == "add":
            return self._make_api_request('post', '/addresses', {
                'user_id': user_id,
                'address_data': address_data or {}
            })
        elif action == "update":
            return self._make_api_request('put', '/addresses', {
                'user_id': user_id,
                'address_data': address_data or {}
            })
        elif action == "delete":
            return self._make_api_request('delete', '/addresses', {
                'user_id': user_id,
                'address_id': kwargs.get('address_id')
            })
        else:
            return {
                "status": "success",
                "message": "Address management options",
                "actions": ["view", "add", "update", "delete"]
            }

    def manage_profile(self, user_id: str, action: str = "view", profile_data: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Manage user profile"""
        if action == "view":
            return self._make_api_request('get', '/profile', {
                'user_id': user_id
            })
        elif action == "update":
            return self._make_api_request('put', '/profile', {
                'user_id': user_id,
                'profile_data': profile_data or {}
            })
        else:
            return {
                "status": "success",
                "message": "Profile management",
                "current_action": "view_profile"
            }

    def manage_notifications(self, user_id: str, action: str = "view", notification_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Manage user notifications"""
        params = {'user_id': user_id}
        if notification_type:
            params['type'] = notification_type

        if action == "view":
            return self._make_api_request('get', '/notifications/settings', params)
        elif action == "enable":
            return self._make_api_request('put', '/notifications/enable', params)
        elif action == "disable":
            return self._make_api_request('put', '/notifications/disable', params)
        else:
            return {
                "status": "success",
                "message": "Notification settings",
                "actions": ["view", "enable", "disable"]
            }

    def apply_coupon(self, user_id: str, coupon_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Apply a coupon code"""
        if coupon_code:
            return self._make_api_request('post', '/coupons/apply', {
                'user_id': user_id,
                'coupon_code': coupon_code
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please provide a coupon code",
                "required_params": ["coupon_code"]
            }

    def remove_coupon(self, user_id: str, coupon_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Remove a coupon code"""
        if coupon_code:
            return self._make_api_request('delete', '/coupons/remove', {
                'user_id': user_id,
                'coupon_code': coupon_code
            })
        else:
            return self._make_api_request('delete', '/coupons/remove-all', {
                'user_id': user_id
            })

    def track_order(self, user_id: str, order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Track an order"""
        if order_id:
            return self._make_api_request('get', '/orders/track', {
                'user_id': user_id,
                'order_id': order_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Which order would you like to track?",
                "required_params": ["order_id"]
            }

    def return_item(self, user_id: str, item_id: Optional[str] = None, order_id: Optional[str] = None, reason: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Return an item"""
        if item_id and order_id:
            return self._make_api_request('post', '/returns/create', {
                'user_id': user_id,
                'item_id': item_id,
                'order_id': order_id,
                'reason': reason or "Not specified"
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and order for return",
                "required_params": ["item_id", "order_id"]
            }

    def exchange_item(self, user_id: str, item_id: Optional[str] = None, order_id: Optional[str] = None, new_item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Exchange an item"""
        if item_id and order_id:
            return self._make_api_request('post', '/exchanges/create', {
                'user_id': user_id,
                'original_item_id': item_id,
                'order_id': order_id,
                'new_item_id': new_item_id
            })
        else:
            return {
                "status": "additional_info_needed",
                "message": "Please specify the item and order for exchange",
                "required_params": ["item_id", "order_id"]
            }

    def view_wishlist(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """View user's wishlist"""
        return self._make_api_request('get', '/wishlist', {
            'user_id': user_id
        })

    def clear_cart(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Clear the entire cart"""
        return self._make_api_request('delete', '/cart/clear', {
            'user_id': user_id
        })

    # === Newly added / expanded lean intent handlers ===

    # Navigation / high-level
    def go_home(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Navigate to home screen"""
        return self._make_api_request('get', '/flow/home', {'user_id': user_id})

    def open_account(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Open account/profile section (alias to manage_profile view)"""
        return self.manage_profile(user_id=user_id, action="view")

    def show_orders(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """List user's orders"""
        return self._make_api_request('get', '/orders', {'user_id': user_id})

    # Product info (additional granularity)
    def show_price(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show only price info"""
        if item_id:
            return self._make_api_request('get', '/items/price', {
                'user_id': user_id,
                'item_id': item_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which item price?",
            "required_params": ["item_id"]
        }

    def show_reviews(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show item reviews"""
        if item_id:
            return self._make_api_request('get', '/items/reviews', {
                'user_id': user_id,
                'item_id': item_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which item reviews?",
            "required_params": ["item_id"]
        }

    def show_similar_items(self, user_id: str, item_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show similar items to a given item"""
        if item_id:
            return self._make_api_request('get', '/items/similar', {
                'user_id': user_id,
                'item_id': item_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which item to find similar ones for?",
            "required_params": ["item_id"]
        }

    # Filters / sorting
    def apply_filter(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Apply search/filter parameters (expects slots)"""
        filters = {k: v for k, v in kwargs.items() if k not in ('user_id', 'intent', 'confidence')}
        if not filters:
            return {
                "status": "additional_info_needed",
                "message": "No filters specified",
                "expected": ["category", "brand", "price_range", "rating", "qualifier", "color", "size"]
            }
        data = {'user_id': user_id, 'filters': filters}
        return self._make_api_request('post', '/items/filter', data)

    def clear_filter(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Clear all active filters"""
        return self._make_api_request('delete', '/items/filters/clear', {'user_id': user_id})

    def sort_items(self, user_id: str, sort_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Sort current item listing"""
        if not sort_key:
            return {
                "status": "additional_info_needed",
                "message": "Which sort key?",
                "expected": ["price_asc", "price_desc", "newest", "popular", "rating_desc"]
            }
        return self._make_api_request('post', '/items/sort', {
            'user_id': user_id,
            'sort_key': sort_key
        })

    # Quantity management (absolute)
    def set_quantity(self, user_id: str, item_id: Optional[str] = None, quantity: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Set absolute quantity for an item (replaces existing)"""
        if item_id and quantity is not None:
            return self._make_api_request('put', '/cart/items', {
                'user_id': user_id,
                'item_id': item_id,
                'quantity': quantity
            })
        return {
            "status": "additional_info_needed",
            "message": "Need item_id and quantity",
            "required_params": ["item_id", "quantity"]
        }

    # Order cancellation (explicit)
    def cancel_order(self, user_id: str, order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Cancel a specific order"""
        if order_id:
            return self._make_api_request('post', '/orders/cancel', {
                'user_id': user_id,
                'order_id': order_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which order to cancel?",
            "required_params": ["order_id"]
        }

    def order_not_arrived(self, user_id: str, order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Report that an order has not arrived"""
        if order_id:
            return self._make_api_request('post', '/orders/support/not-arrived', {
                'user_id': user_id,
                'order_id': order_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which order has not arrived?",
            "required_params": ["order_id"]
        }

    def show_order_status(self, user_id: str, order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show status for a specific order"""
        if order_id:
            return self._make_api_request('get', '/orders/status', {
                'user_id': user_id,
                'order_id': order_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which order status?",
            "required_params": ["order_id"]
        }

    def refund_status(self, user_id: str, order_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Check refund status"""
        if order_id:
            return self._make_api_request('get', '/orders/refund-status', {
                'user_id': user_id,
                'order_id': order_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Which order refund status?",
            "required_params": ["order_id"]
        }

    # Addresses granular
    def show_addresses(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self.manage_address(user_id=user_id, action="view")

    def add_address(self, user_id: str, address_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if address_data:
            return self.manage_address(user_id=user_id, action="add", address_data=address_data)
        return {
            "status": "additional_info_needed",
            "message": "Address data required",
            "required_params": ["address_data"]
        }

    def remove_address(self, user_id: str, address_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if address_id:
            return self.manage_address(user_id=user_id, action="delete", address_id=address_id)
        return {
            "status": "additional_info_needed",
            "message": "Address id required",
            "required_params": ["address_id"]
        }

    def set_default_address(self, user_id: str, address_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if address_id:
            return self._make_api_request('put', '/addresses/default', {
                'user_id': user_id,
                'address_id': address_id
            })
        return {
            "status": "additional_info_needed",
            "message": "Address id required to set default",
            "required_params": ["address_id"]
        }

    # Notifications granular
    def enable_order_updates(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('put', '/notifications/order-updates/enable', {'user_id': user_id})

    def disable_order_updates(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('put', '/notifications/order-updates/disable', {'user_id': user_id})

    def enable_price_alert(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('put', '/notifications/price-alert/enable', {'user_id': user_id})

    def disable_price_alert(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('put', '/notifications/price-alert/disable', {'user_id': user_id})

    # Support
    def start_live_chat(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('post', '/support/live-chat/start', {'user_id': user_id})

    def show_faqs(self, user_id: str, **kwargs) -> Dict[str, Any]:
        return self._make_api_request('get', '/support/faqs', {'user_id': user_id})
