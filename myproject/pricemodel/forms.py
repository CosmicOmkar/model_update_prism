from django import forms


class URL(forms.Form):
    url = forms.CharField(label='URL', max_length=200)  # You can adjust the max_length as needed
