from django.http import JsonResponse
from .objects import SystemMaintenance

# This function should be defined for every app
# in the view. For example:
# from ...core.apps_general_functions import activate_obj_function
def activate_obj_function(request, add_dic=None):
    try:
        # print("activate_obj_function 9020")
        # print(add_dic)
        if add_dic:
            app_ = add_dic["app"]
            obj_ = add_dic['obj']
            fun_ = add_dic['fun']
            params = add_dic["params"]
            try:
                obj_param = add_dic['obj_param']
                if obj_param == "":
                    obj_param = None
            except Exception as ex:
                print("9044: "+str(ex))
        else:
            dic_ = request.POST.get('dic')
            # print('9080-1 dic_ activate_obj_function dic=', '\n', dic_, '\n', '-'*10)
            dic_ = eval(dic_)
            app_ = dic_["app"]
            obj_ = dic_['obj']
            fun_ = dic_['fun']
            params = dic_["params"]
            # print("9047:\nparams:\n", params)
            obj_param = None
            try:
                obj_param = dic_['obj_param']
                if obj_param == "":
                    obj_param = None
            except Exception as ex:
                pass
                # print("9045: "+str(ex))
        s = 'from ..'
        if app_ != "corporatevaluation" and app_ != "core":
            s += 'acapps.'
        s += app_+'.objects import '+obj_
        # print('9081 dic_ activate_obj_function s=', '\n', s, '\n', '-'*10)
        try:
            exec(s)
        except Exception as ex:
            print("9077-77 core view activate_obj_function: "+str(ex))

        # print('9082-1 dic_ activate_obj_function params=', '\n', params, '\n', '-'*10)
        # print('9082-2 dic_ activate_obj_function params=', '\n', obj_param, '\n', '-'*10)

        if obj_param:
            s_ = obj_+'(obj_param).' + fun_ + '(params)'
        else:
            s_ = obj_+'().' + fun_ + '(params)'
        # print('9084 dic_ activate_obj_function s_=', '\n', s_, '\n', '-'*10)

        try:
            print(s_)
            dic = eval(s_)
            # print(dic)
        except Exception as ex:
            print("9088-8 core view activate_obj_function: "+str(ex))
        # print(dic)

        return JsonResponse({'status': 'ok', 'result': dic})
    except Exception as ex:
        print("9090-9 core view activate_obj_function: "+str(ex))
        return JsonResponse({'status': 'ko: activate_obj_function'})


# def activate_obj_function(request):
#     try:
#         dic_ = request.POST.get('dic')
#         dic_ = eval(dic_)
#         # print('9080 dic_ activate_obj_function dic=', '\n', dic_, '\n', '-'*10)
#         app_ = dic_["app"]
#         obj_ = dic_['obj']
#         fun_ = dic_['fun']
#         s = 'from ..'
#         if app_ != "corporatevaluation" and app_ != "core":
#             s += 'acapps.'
#         s += app_+'.objects import '+obj_
#         # print('9081 dic_ activate_obj_function s=', '\n', s, '\n', '-'*10)
#         exec(s)
#         params = dic_["params"]
#         # print('9082 dic_ activate_obj_function params=', '\n', params, '\n', '-'*10)
#         s_ = obj_+'().' + fun_ + '(params)'
#         # print('9084 dic_ activate_obj_function s_=', '\n', s_, '\n', '-'*10)
#         try:
#             dic = eval(s_)
#             # print(dic)
#         except Exception as ex:
#             print(ex)
#         return JsonResponse({'status': 'ok', 'result': dic})
#     except Exception as ex:
#         # print(ex)
#         return JsonResponse({'status': 'ko: activate_obj_function'})
