# tests/test_ecommerce_integration.py
import pytest
# import json
import responses
from src.utils.ecommerce_integration import EcommerceIntegration

class TestEcommerceIntegration:
    @pytest.fixture
    def ecommerce(self):
        """Create EcommerceIntegration instance for testing"""
        return EcommerceIntegration(
            api_key="test_key",
            base_url="https://test-api.example.com"
        )

    @responses.activate
    def test_add_to_cart(self, ecommerce):
        """Test add to cart functionality"""
        # Mock API response
        responses.add(
            responses.POST,
            "https://test-api.example.com/cart/items",
            json={"status": "success", "message": "Item added to cart"},
            status=200
        )

        # Test the function
        result = ecommerce.add_to_cart(
            user_id="test_user",
            item_id="item123",
            quantity=2
        )

        assert result["status"] == "success"
        assert result["message"] == "Item added to cart"

    @responses.activate
    def test_checkout(self, ecommerce):
        """Test checkout functionality"""
        # Mock API response
        responses.add(
            responses.POST,
            "https://test-api.example.com/checkout/start",
            json={"status": "success", "checkout_id": "checkout123"},
            status=200
        )

        # Test the function
        result = ecommerce.checkout(user_id="test_user")

        assert result["status"] == "success"
        assert "checkout_id" in result

    def test_execute_action_valid_intent(self, ecommerce):
        """Test execute_action with valid intent"""
        # Mock the add_to_cart function to avoid API calls
        original_add_to_cart = ecommerce.add_to_cart
        ecommerce.add_to_cart = lambda user_id, **kwargs: {"status": "success", "message": "Test success"}

        try:
            result = ecommerce.execute_action(
                intent="add_to_cart",
                user_id="test_user",
                confidence=0.95,
                item_id="item123"
            )

            assert result["status"] == "success"
            assert result["intent"] == "add_to_cart"
            assert result["confidence"] == 0.95

        finally:
            # Restore original function
            ecommerce.add_to_cart = original_add_to_cart

    def test_execute_action_low_confidence(self, ecommerce):
        """Test execute_action with low confidence"""
        result = ecommerce.execute_action(
            intent="add_to_cart",
            user_id="test_user",
            confidence=0.5  # Below threshold
        )

        assert result["status"] == "low_confidence"
