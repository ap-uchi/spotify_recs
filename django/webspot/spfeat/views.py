from django.shortcuts import render
from django.http import HttpResponse

from .models import MarkdownContent
# Create your views here.

def index(request):
    return HttpResponse("We are at the spfeat index.")

def markdown_content_view(request):
    markdown_content = MarkdownContent.objects.first()
    context = {"markdown_content": markdown_content}
    return render(
        request,
        "spfeat/markdown_content.html",
        context=context
    )