
from django.urls import path
from .views import (home, app, update_field_model, post_ajax_create_action, activate_function, update_field_model_by_id,
                    get_data_link, get_data_json_link,
                    activate_obj_function, upload_file,
                    general_data, logmein, delete_record_from_table

                    # , get_adjective_link
                    )

app_name = "core"

urlpatterns = [
    path('', home, name='home'),
    path('app/<str:app_name>/', app, name='app'),
    path('activate_obj_function/', activate_obj_function, name='activate_obj_function'),
    path('upload_file/', upload_file, name='upload_file'),
    path('general_data/', general_data, name='general_data'),
    path('logmein/', logmein, name='logmein'),
    path('delete_record_from_table/', delete_record_from_table, name='delete_record_from_table'),

    path('post_ajax_create_action', post_ajax_create_action, name='post_ajax_create_action'),
    path('update_field_model', update_field_model, name='update_field_model'),
    path('activate_function', activate_function, name='activate_function'),
    path('update_field_model_by_id', update_field_model_by_id, name='update_field_model_by_id'),
    path('get_data_link', get_data_link, name='get_data_link'),
    path('get_data_json_link', get_data_json_link, name='get_data_json_link'),
    # path('get_adjective_link', get_adjective_link, name='get_adjective_link'),

]

