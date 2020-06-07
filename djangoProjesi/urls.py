from django.conf.urls import url
from ilkUygulama import views

urlpatterns = [
    url(r'^$',views.AnasayfaSayfaGorunumu.as_view()),
    url(r'^hakkımda/$',views.HakkimdaSayfaGorunumu.as_view()),
    url(r'^iletişim/$',views.IletisimSayfaGorunumu.as_view()),
]