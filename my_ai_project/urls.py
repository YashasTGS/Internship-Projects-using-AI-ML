from django.contrib import admin
from django.urls import path
from main_app import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Authentication
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # Dashboard (MAIN PAGE with all projects)
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # Project Routes (all connected to one page)
    path('plant/', views.plant, name='plant'),
    path('city/', views.city, name='city'),
    path('detect/', views.detect, name='detect'),
]