from django.test import TestCase, Client
from django.urls import reverse


class DrBaranesTest(TestCase):
    def test_should_respond_only_for_example_a(self):
        client = Client(HTTP_HOST="DrBaranes.com")
        view = reverse("core:home")
        response = client.get(view)
        self.assertEqual(response.status_code, 200)
