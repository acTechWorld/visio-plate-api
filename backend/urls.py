from django.contrib import admin
from django.urls import path
from recognition.views import PlateRecognitionAPI

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/plate/', PlateRecognitionAPI.as_view()),
]
