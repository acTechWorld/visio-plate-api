from rest_framework import serializers


class PlateUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
