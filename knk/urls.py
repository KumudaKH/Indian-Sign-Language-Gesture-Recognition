"""knk URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
	https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
	1. Add an import:  from my_app import views
	2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
	1. Add an import:  from other_app.views import Home
	2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
	1. Import the include() function: from django.urls import include, path
	2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.urls import path
from aud2gest import views as aud_view
from user import views as user_view
from gest2aud import views as gest_view

urlpatterns = [ 
	# Main landing page
	path('', aud_view.index, name="home"),
	
	# Audio to Gesture (aud2gest) app routes
	path('home/', aud_view.home, name='aud_home'),
	path('index/', aud_view.index, name='aud_index'),
	path('instruction/', aud_view.instruction, name='instruction'),
	path('api/predict-frame/', aud_view.predict_frame, name='predict_frame'),
	
	# User authentication routes
	path('login/', user_view.login_user, name='login'),
	path('register/', user_view.register, name='register'),
	path('logout/', user_view.logout_user, name='logout'),
	
	# Admin
	path('admin/', admin.site.urls),
	
	# Gesture to Audio (gest2aud) app routes
	path('webcam/', gest_view.take_snaps, name="webcam"),
	path('api/test-connection/', gest_view.test_connection, name='test_connection'),
	path('api/predict-webcam/', aud_view.predict_frame, name='predict_webcam_frame'),
	path('gest_keyboard/', gest_view.gest_keyboard, name="gest_keyboard"),
	path('emergency/', gest_view.emergency, name='emergency')

]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
