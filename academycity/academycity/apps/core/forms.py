from django import forms
from .models import AdjectivesValues


field2 = forms.ModelChoiceField(queryset=AdjectivesValues.adjectives.get_queryset('field2'))



