
import os
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # academycity/academycity
# print(BASE_DIR)

WEB_DIR = os.path.dirname(BASE_DIR)                                      # academycity
DATA_DIR = os.path.dirname(os.path.dirname(__file__))                    # academycity/academycity/setting
OBJECTS_PATH = 'objects'

STATIC_URL = '/static/'
MEDIA_URL = '/media/'

STATICFILES_DIRS = (os.path.join(BASE_DIR, 'static'),)

MEDIA_ROOT = os.path.join(BASE_DIR, "media")
STATIC_ROOT = os.path.join(WEB_DIR, "static")

DATA_ROOT = os.path.join(BASE_DIR, "Data")
LOCALE_PATHS = (os.path.join(BASE_DIR, 'locale'), )


#
# # Build paths inside the project like this: BASE_DIR / 'subdir'.
# BASE_DIR = Path(__file__).resolve().parent.parent
#
# MEDIA_ROOT = os.path.join(BASE_DIR, "media")
# WEB_ROOT = os.path.join(BASE_DIR, "academycity")
# STATIC_ROOT = os.path.join(WEB_ROOT, "static")
#
# print("SR", STATIC_ROOT)
#
# # Quick-start development settings - unsuitable for production
# # See https://docs.djangoproject.com/en/dev/howto/deployment/checklist/


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-4a$rurgz27kqp-uo*1#s0kf3mslybzhwyh-nok_5x0#m6ag-f8'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'academycity.apps.core',
    'academycity.apps.acapps.ml',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'academycity.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'academycity.wsgi.application'


# Database
# https://docs.djangoproject.com/en/dev/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR + '/db.sqlite3',
    }
}

# Password validation
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]



LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

