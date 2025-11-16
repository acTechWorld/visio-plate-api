from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .serializers import PlateUploadSerializer
from .utils import detect_plate_yolo


class PlateRecognitionAPI(APIView):
    def post(self, request):
        serializer = PlateUploadSerializer(data=request.data)

        if serializer.is_valid():
            image = serializer.validated_data['image']

            # save temp file
            temp_path = f"/tmp/{image.name}"
            with open(temp_path, "wb+") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            result = detect_plate_yolo(temp_path)

            return Response({
                "plate_number": result
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
