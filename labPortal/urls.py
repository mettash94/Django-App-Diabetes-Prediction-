from django.urls import path
from labPortal import views

urlpatterns = [
    path('home/', views.home, name="home"),
    path('home/dashboard/', views.dashboard, name="dashboard"),
    path('home/dashboard/prediction', views.prediction, name="prediction")
]
