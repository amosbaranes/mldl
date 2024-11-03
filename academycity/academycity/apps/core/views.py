from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, authenticate, logout
from django.utils.translation import get_language, gettext_lazy as _
from django.core.management import call_command
from django.apps import apps
from django.http import JsonResponse
from django.conf import settings
import os
from django.utils.dateparse import parse_date
from django.contrib.auth.models import User
from .AdvancedTabes import AdvancedTabs
from ..core.utils import log_debug
from django.urls import reverse
from .apps_general_functions import activate_obj_function
from django.db.models.fields.related import ForeignKey, ManyToManyField, ManyToManyRel

from django.http import HttpResponse
import json
import pandas as pd
import numbers

def home(request):
    title = _('Core App')


    return render(request, 'core/home.html', {'title': title})

def app(request, app_name):
    app_ = "core"
    app_activate_function_link_ = reverse(app_+':activate_obj_function', kwargs={})
    return render(request, app_+'/advanced_tabs.html', {"atm_name": "core_"+app_name+"_tm", "app": app_,
                                               "app_activate_function_link": app_activate_function_link_,
                                               "company_obj_id": 0, "title": "Core"}
                  )

def post_ajax_create_action(request):
    try:
        obj = None
        if request.POST.get('model') and request.POST.get('model') != "":
            app_ = request.POST.get('app')
            model_ = request.POST.get('model')
            model = apps.get_model(app_label=app_, model_name=model_)
            try:
                obj_slug = request.POST.get('obj_slug')
                # print(obj_slug)
                obj = model.objects.filter(translations__language_code=get_language()).filter(translations__slug=obj_slug).all()[0]
            except Exception as er:
                pkey_ = request.POST.get('pkey')
                obj = model.objects.get(id=pkey_)

        verb_ = request.POST.get('verb')
        create_action(request.user, verb_, obj)
    except Exception as err:
        JsonResponse({'status': 'ko'})
    return JsonResponse({'status': 'ok'})

def update_field_model(request):
    app_ = request.POST.get('app')
    model_ = request.POST.get('model')

    pkey_ = request.POST.get('pkey')
    value_ = request.POST.get('value')
    field_ = request.POST.get('field')
    type_ = request.POST.get('type')
    # print('1-'*20)
    # print(value_)
    # print(model_)
    # print(pkey_)
    # print('1-'*20)
    # print('2-'*20)
    # print(field_)
    # print('2-'*20)
    model = apps.get_model(app_label=app_, model_name=model_)
    try:
        obj_slug = request.POST.get('obj_slug')
        # print(obj_slug)
        obj = model.objects.filter(translations__language_code=get_language()).filter(translations__slug=obj_slug).all()[0]
    except Exception as er:
        obj = model.objects.get(id=pkey_)
    # print(obj)
    # print('3-'*20)
    # print(type_)
    # print('4-'*20)
    if type_ == "checkbox":
        if value_ == 'true':
            value_ = True
        else:
            value_ = False
    elif type_ == "date":
        value_ = parse_date(value_)
    elif type_ == "multiple_select":
        value_ = value_.split(",")
        for k in obj.instructors.all():
            obj.instructors.remove(k)
        for i in value_:
            u_ins = User.objects.get(id=int(i))
            obj.instructors.add(u_ins)
        return JsonResponse({'status': 'ok'})
    try:
        # print(value_)
        s = 'obj.' + field_ + ' = value_'
        # print(s)
        # print('-'*30)
        # print(s)
        # print('-'*30)
        exec(s)
        obj.save()
        if model_ == "Game" and field_ == "number_of_periods":
            obj.get_schedule_period_dates
        return JsonResponse({'status': 'ok'})
    except Exception as e:
        # print('-'*30)
        # print("err")
        # print(e)
        # print("err")
        # print('-'*30)
        pass
        return JsonResponse({'status': 'ko'})

def create_db_backup():
    call_command('dbbackup', compress=True, clean=True)

def clean_registrations(request):
    clean_accounting_registrations()

def activate_function(request):
    # var dic_ = {"obj": "AdvancedTabs", "atm": atm_.my_name, "app": atm_.my_app, "fun": "add_tab",
    #             "params": {"tab_name": tab_name_}}
    try:
        dic_ = request.POST["dic"]
        dic_ = eval(dic_)
        # print("-"*50, "\n", "core: 9111-1: dic\n", dic_, "\n", "-"*50, "\n")
        #
        obj_ = dic_["obj"]
        atm_ = dic_["atm"]
        app_ = dic_["app"]
        fun_ = dic_["fun"]
        params_ = dic_["params"]
        # print("core\n9005: params_\n", "-"*50, params_, '\n', "-"*50)
        s = obj_ + "('"+atm_+"', '"+app_+"')." + fun_ + "(params_)"

        # print("core\n", "9010: s=\n", s, "\n", "-"*50)

        result = eval(s)
        # print('-3'*20)
        dic = {'status': 'ok', 'result': result}
        # print("core\n", "9015: dic=\n", dic, "\n", "-"*50)
    except Exception as ex:
        print("90265-55 Error core activate_function: "+str(ex))
        dic = {'status': 'ko', 'result': "{}"}
    return JsonResponse(dic)

def update_field_model_by_id(request, foreign=None):
    log_debug("update_field_model_by_id 0")
    dic_ = request.POST["dic"]
    # print('90155-100 core update_field_model_by_id: dic_\n', '-'*50,"\n", dic_, "\n", '-'*50)
    # log_debug(dic_) # size is 1024 chars
    dic_ = eval(dic_)
    # print('90155-100-1 core update_field_model_by_id: dic_\n', '-'*50,"\n", dic_, "\n", '-'*50)
    element_id_ = ""
    try:
        element_id_ = dic_["element_id"]
    except Exception as ex:
        pass

    try:
        app_ = dic_['app']
        # print(dic_['model'])
        model_ = dic_['model']
        pkey_ = dic_['pkey']
        fields_ = dic_['fields']
        type_ = dic_['type']
        company_obj_id_ = dic_['company_obj_id']
        parent_model_ = dic_['parent_model']
        parent_model = apps.get_model(app_label=app_, model_name=app_+"web")
        company_obj = parent_model.objects.get(id=company_obj_id_)
    except Exception as ex:
        log_debug("Error 2000:" + str(ex))

    try:
        model = apps.get_model(app_label=app_, model_name=model_)
    except Exception as ex:
        log_debug("Error 2001:" + str(ex))
        return JsonResponse({'status': 'ko'})
    p_key_field_name = ""

    # print("9087-67 pkey_", pkey_)

    if pkey_ == "new":
        s = ""
        if model.model_field_exists(app_+'_web') and isinstance(model._meta.get_field(app_+'_web'), ForeignKey):
            # print("it is a foreign key")
            s = "model.objects.create("
            s += app_ + '_web=company_obj '
        # print("9066-66:\n" + s)
        # print(parent_model_)
        if parent_model_ != "":
            # print('9040\n', s)
            parent_pkey_ = dic_['parent_pkey']
            parent_model__ = apps.get_model(app_label=app_, model_name=parent_model_)
            parent_model_fk_name = parent_model_[:-1]
            parent_obj__ = parent_model__.objects.get(id=parent_pkey_)
            if s == "":
                s = "model.objects.create("
            else:
                s += ", "
            s += parent_model_fk_name + '=parent_obj__'
        try:
            # print('9045-0')
            space_ = ", "
            if s == "":
                s = "model.objects.create("
                space_ = ""
            # print("9045-1 ", s)

            foreign_keys = dic_["foreign_keys"]
            # print("9046 ", foreign_keys, "9046 ")

            for k in foreign_keys:
                t_model = apps.get_model(app_label=app_, model_name=foreign_keys[k]["foreign_table"])
                p_key_field_name = t_model._meta.pk.name
                try:
                    # print(k, foreign_keys[k]["value"])
                    n_p_key = int(foreign_keys[k]["value"])
                    # print('t_model.objects.get('+p_key_field_name+'=n_p_key)')
                    obj__ = eval('t_model.objects.get('+p_key_field_name+'=n_p_key)')
                    # print("=6"*10)
                    # print(obj__)
                    # print("=6"*10)
                except Exception as ex:
                    print("err 800-1: "+str(ex))
                try:
                    myVars = vars()
                except Exception as ex:
                    print("err 800: "+str(ex))
                try:
                    myVars[k] = obj__
                except Exception as ex:
                    print("err 900: "+str(ex))
                    log_debug("err 900: "+str(ex))
                s += space_+k+'='+k
                space_ = ", "
        except Exception as ex:
            pass
            # print("7010-1 save with fkey error " + str(ex))
        s += ')'

        # print("9066-661:\n" + s)

        try:
            # print("9055-89\n" + s, "\n", "-"*30)
            # print(model)
            obj = eval(s)
        except Exception as ex:
            print("error701-701-1 "+str(ex))
        try:
            p_key_field_name = model._meta.pk.name
            if model.model_field_exists(app_ + '_web') and isinstance(model._meta.get_field(app_ + '_web'), ManyToManyField):
                s = "obj."+app_ + '_web.add(company_obj)'
                # print("9035\n", s)
                eval(s)
                # print("9052")
                obj.save()
                print("9055")
            # print("9055-89\n", "-"*30)
        except Exception as ex:
            print("error701-701 "+str(ex))
            log_debug("error701 "+str(ex))
    else:
        try:
            obj_slug = request.POST.get('obj_slug')
            # print(obj_slug)
            obj = \
            model.objects.filter(translations__language_code=get_language()).filter(translations__slug=obj_slug).all()[0]
        except Exception as er:
            log_debug("update_field_model_by_id 1311")
            p_key_field_name = model._meta.pk.name
            obj = eval('model.objects.get('+p_key_field_name+'=pkey_)')
            log_debug("update_field_model_by_id 1322")
            # print("obj: ", obj)
            # obj = model.objects.get(id=pkey_)

    # print('9088-1  obj')
    # print(obj)
    # print('9088-1  obj')

    try:
        for f in fields_:
            value_ = fields_[f]
            # print("9071 ", f, value_)
            try:
                if isinstance(model._meta.get_field(f), ManyToManyRel):
                    l_ = value_.split(",")
                    for z in l_:
                        print(z)
                        model__ = apps.get_model(app_label=app_, model_name=f.split("_")[1])
                        model__v = model__.objects.get(id=int(z))
                        print(model__v)
                        try:
                            print("model__v." + f.split("_")[0] + "_web.add(obj)")
                            exec("model__v." + f.split("_")[0] + "_web.add(obj)")
                            model__v.save()
                        except Exception as ex:
                            print("Error 9072 " + ex)

                    return JsonResponse({'status': 'ok', "record_id": obj.id})
            except Exception as ex:
                print("Error 4500-1 core update_field_model_by_id: "+str(ex))

            # print("f", f, "fields_[f]", fields_[f], type_)
            if type_ == "checkbox":
                if value_ == 'true':
                    value_ = True
                elif value_ == 'false':
                    value_ = False
            elif type_ == "date":
                try:
                    v_ = parse_date(value_)
                    if v_:
                        value_ = v_
                except Exception as exx:
                    print(exx)
            elif type_ == "multiple_select":
                # need to fix this part
                # need to fix this part
                value_ = value_.split(",")
                for k in obj.instructors.all():
                    obj.instructors.remove(k)
                for i in value_:
                    u_ins = User.objects.get(id=int(i))
                    obj.instructors.add(u_ins)
                # need to fix this part
                # need to fix this part
                return JsonResponse({'status': 'ok'})
            try:
                try:
                    foreign_keys = dic_["foreign_keys"]
                    # print(foreign_keys, f)
                    model_f = apps.get_model(app_label=app_, model_name=foreign_keys[f]["foreign_table"])
                    # print(model_f)
                    # print("9011 ", value_)
                    value_ = model_f.objects.get(id=value_)
                    # print("9012 ", value_)
                except Exception as ex:
                    pass
                    # print("Error 600: "+str(ex))
                setattr(obj, f, value_)
            except Exception as ex:
                print("Error 800: "+str(ex))
        try:
            obj.save()
        except Exception as eex:
            print("Error 850: "+str(eex))

        if p_key_field_name != "":
            # print(('obj.' + p_key_field_name+".id"))
            # ooo = eval('obj.' + p_key_field_name+".id")
            # print(ooo)
            # isinstance(obj_id, User)
            try:
                obj_id = eval('obj.' + p_key_field_name)
                obj_id = int(obj_id)
            except Exception as ex:
                obj_id = eval('obj.' + p_key_field_name + '.id')
        ret = {'status': 'ok', "record_id": obj_id, "element_id":element_id_}
        # print(ret)

        return JsonResponse(ret)
    except Exception as e:
        log_debug("erro600 " + str(e))
        return JsonResponse({'status': 'ko'})

def get_data_link(request):
    errors = ""
    dic_ = request.POST["dic"]
    try:

        # log_debug("get_data_link 99999: "+dic_)
        # print("get_data_link 99999: \n", dic_, "\n")
        dic_ = eval(dic_)
        # print("\nget_data_link 99999-2: \n", dic_, "\n")
    except Exception as ex:
        print("error 4562-22-1",str(ex))


    # errors += "1 "
    # log_debug("get_data_link 99999: "+errors)
    parent_model_fk_name=""
    try:
        parent_model_fk_name=dic_["parent_model_fk_name"]
    except Exception as ex:
        pass

    parent_model_fk_name=""
    try:
        parent_model_fk_name=dic_["parent_model_fk_name"]
    except Exception as ex:
        pass

    # print("AAAAA5", type(dic_), dic_)
    # print("parent_model_fk_name\n", parent_model_fk_name, "\nparent_model_fk_name")
    # print("AAAAAAAA777", type(dic_), dic_)

    multiple_select_fields = None
    if "multiple_select_fields" in dic_:
        if len(dic_["multiple_select_fields"]) > 0:
            multiple_select_fields = dic_["multiple_select_fields"]
    app_ = dic_['app']
    model_ = dic_['model']
    # print("model_: "+model_)
    if model_ == "":
        dic = {'status': 'ko', "dic": {}}
        return JsonResponse(dic)

    model = apps.get_model(app_label=app_, model_name=model_)
    # for testing only --
    # model__ = apps.get_model(app_label=app_, model_name="XBRLDimCompany")
    # df = pd.DataFrame(model__.objects.all().values())
    # print(df)
    # for index, row in df.iterrows():
    #     print(index)
    #     # print(row, "\n", row["full_name"])
    #     print(row["sic_code"])
    # for testing only --
    p_key_field_name = model._meta.pk.name
    if p_key_field_name not in dic_["fields"]:
        dic_["fields"].insert(0, p_key_field_name)
    fields_str = '"'
    for f in dic_["fields"]:
        try:
            if f != "":
                exec(f + ' = []')
                fields_str += f + '","'
        except Exception as ex:
            print("error 4000-1: "+str(ex))
    fields_str = fields_str[:len(fields_str)-2]
    # print("9030","\n", fields_str,"\n","=2"*50)
    # fields_ = model._meta.get_fields(include_parents=True, include_hidden=True)
    # print(fields_)
    number_of_rows_ = 2
    try:
        number_of_rows_ = dic_['number_of_rows']
        number_of_rows_ = int(number_of_rows_)
    except Exception as ex:
        pass
        # print(ex)
    parent_id_ = -1
    try:
        parent_id_ = int(dic_['parent_id'])
    except Exception as ex:
        # print("error 500 "+str(ex))
        pass
    try:
        company_obj_id_ = dic_['company_obj_id']
    except Exception as ex:
        print("error 440: "+str(ex))
    filters = dic_['filters']
    if len(dic_['order_by']) > 0:
        order_by = dic_['order_by']
    else:
        order_by = ""

    # print(" company_obj_id_", company_obj_id_)

    if company_obj_id_ != "-1" and company_obj_id_ != -1:
        # log_debug("get_data_link company_obj_id_: "+str(company_obj_id_))
        s = 'model.objects'
        s_ = ''
        try:
            parent_model = apps.get_model(app_label=app_, model_name=app_+"web")
            # print(parent_model)
            company_obj = parent_model.objects.get(id=company_obj_id_)
            if model.model_field_exists(app_+'_web') and isinstance(model._meta.get_field(app_+'_web'),
                                                                    ForeignKey):
                s_ += app_ + '_web=company_obj '
            if parent_id_ > -1:
                parent_model_ = dic_['parent_model']
                parent_pkey_ = parent_id_
                parent_model__ = apps.get_model(app_label=app_, model_name=parent_model_)

                ss_=parent_model_
                if ss_[len(ss_)-1] == "s":
                    ss_ = parent_model_[:-1]
                if parent_model_fk_name == "":
                    parent_model_fk_name = ss_
                pk = parent_model__._meta.pk.name
                parent_obj__ = eval('parent_model__.objects.get('+pk+'=parent_pkey_)')
                if s_ != '':
                    s_ += ', '
                s_ += parent_model_fk_name+'=parent_obj__'
                # print("sss1", "\n", s, "\n", sss1")
        except Exception as ex:
            print(ex)
        if s_ != '':
            s += '.filter('+s_+')'
        # print('s00111\n', s, '\ns00111')
    else:
        if parent_id_ > -1:
            parent_model_ = dic_['parent_model']
            parent_pkey_ = parent_id_
            parent_model__ = apps.get_model(app_label=app_, model_name=parent_model_)

            if parent_model_fk_name == "":
                parent_model_fk_name = parent_model_[:-1]

            # parent_obj__ = parent_model__.objects.get(id=parent_pkey_)

            parent_obj__ = eval('parent_model__.objects.get('+parent_model__._meta.pk.name+'=parent_pkey_)')

            # print("parent_obj__\n", parent_obj__)

            s = 'model.objects.filter(' + parent_model_fk_name+'=parent_obj__)'
        else:
            s = 'model.objects'
        # print('90500 s '+s)
    # print("9030-2\n", s)
    try:
        # print("filters\n", filters, "\n")
        for f in filters:
            filter_field_ = f  # filters[f]["filter_field"] #
            filter_value_ = str(filters[f]["value"])
            filter_field_a = ""
            try:
                filter_field_a = str(filters[f]["field"])
            except Exception as exx:
                pass
            foreign_table_ = ""
            try:
                foreign_table_ = filters[f]["foreign_table"]
            except Exception as exx:
                pass
            if filter_value_ != "":
                # print(foreign_table_)
                if foreign_table_ != "":
                    # print(1111111111)
                    if filter_field_a != "":
                        # need need need to check this one. I changed it and it might have effect on other reports
                        filter_field_ = filter_field_a
                    # print(filter_field_)
                    # print(111122222)
                    # f__ = model._meta.get_field(filter_field_)
                    # print(f__)
                    # t__ = f__.get_internal_type()
                    # print(t__)
                    # print(str(t__))
                    # s += '.filter('+foreign_table_+'__'+filter_field_+'='+filter_value_+')'
                    # if str(t__)=="AutoField":
                    #     print(3333333)
                    index = filter_field_.find("id")
                    if index != -1:
                        s += '.filter(' + foreign_table_ + '__' + filter_field_ + '=' + filter_value_ + ')'
                        # s += '.filter(' + foreign_table_ + '__' + filter_field_ + '=' + filter_value_ + ')'
                    else:
                        # print(44444555)
                        s += '.filter(' + foreign_table_ + '__' + filter_field_ + '__icontains="'+filter_value_+'")'
                        # s += '.filter(' + foreign_table_ + '__' + filter_field_ + '__icontains="'+filter_value_+'")'
                else:
                    # print(22222222222)
                    if filter_field_ == "id":
                        #s += '.filter('+filter_field_+'__icontains='+filter_value_+')'
                        s += '.filter('+filter_field_+'='+filter_value_+')'
                    else:
                        ss_=""
                        if filter_value_[0] == ">":
                            filter_value_ = filter_value_[1:]
                            ss_="__gt"
                            if filter_value_[0] == "=":
                                filter_value_ = filter_value_[1:]
                                ss_ += "e"
                        if filter_value_[0] =="<":
                            filter_value_ = filter_value_[1:]
                            ss_="__lt"
                            if filter_value_[0] == "=":
                                filter_value_ = filter_value_[1:]
                                ss_ += "e"
                        # print(ss_)
                        if filter_value_.isnumeric():
                            s += '.filter('+filter_field_+ss_+'='+filter_value_+')'
                        else:
                            s += '.filter('+filter_field_+'__icontains="'+filter_value_+'")'
                        # print("-"*10)
                        # print(s)
                        # print("-"*10)
        # print("9030-22-1")
        # print(s)
        # print("9030-22-2")
        n_ = -1
        try:
            primary_key_list_filter_ = dic_["primary_key_list_filter"]
            n_ = len(primary_key_list_filter_)
            if primary_key_list_filter_ and n_ > 0:
                s += '.filter('+p_key_field_name+'__in=primary_key_list_filter_)'
        except Exception as ex:
            ("Error 90855-23 "+str(ex))

        # print("9030-221")
        # print(order_by)
        # print("9030-221-1")

        if order_by != "":
            s += '.order_by("'+order_by["field"]+'")'
            if order_by["direction"] == "descending":
                s += '.reverse()'
        # print("9030-222")
        if multiple_select_fields:
            ss__ = s+'.all()[:number_of_rows_]'
            # print('ss__ for data__')
            # print(ss__)
            # print('ss__')
            data__ = eval(ss__)

        # print("9030-223")
        s += '.all()[:number_of_rows_].values('+fields_str+')'

        # print("="*100, '\ns111-1 for d_data\n', "\ns=", number_of_rows_, s, "=\n", "="*100, "\n")

        log_debug("get_data_link 99999: s="+s)
        d_data = eval(s)
        # print("d_data\n", d_data)
    except Exception as ex:
        print("3030-1 core views error 300 "+str(ex))
        log_debug("3030-1 core views error 300 "+str(ex))
        # pass
    dic = {}
    if multiple_select_fields:
        for z in multiple_select_fields:
            dic[z] = []
            for q in data__:
                model_z_name = z+"s"
                model_z = apps.get_model(app_label=app_, model_name=model_z_name)
                df = pd.DataFrame(model_z.objects.all().values())
                p_key_field_z_name = model_z._meta.pk.name
                if p_key_field_z_name != "id":
                    p_key_field_z_name = p_key_field_z_name + "_id"
                qs = eval('q.'+z+'.all()')
                s = ""
                for q_ in qs:
                    if s != "":
                        s += ","
                    s += str(eval('q_.'+p_key_field_z_name))
                dic[z].append(s)
    # print("3030-150 core views get_data_link ")
    log_debug("3032-152 core views get_data_link ")
    try:
        for q in d_data:
            for f in dic_["fields"]:
                if f != "":
                    # print(f+'.append(q[\''+f+'\'])')
                    kk__ = q[f]
                    if (not (isinstance(q[f], float) or isinstance(q[f], int))) and isinstance(kk__, numbers.Number):
                        # print(isinstance(q[f], numbers.Number))
                        kk__ = float(kk__)
                    # eval(f+'.append(q[\''+f+'\'])')
                    eval(f+'.append(kk__)')
                    # print(eval(f))
        for ff in dic_["fields"]:
            if ff != "":
                dic[ff] = eval(ff)
    except Exception as ex:
        print("3031-160 core views error 330 "+str(ex))
        log_debug("3032-165 core views error 333 "+str(ex))
    dic["pkf_name"] = p_key_field_name
    # print(dic, "=2"*50)
    dic = {'status': 'ok', "dic": dic}
    # print('core view 9055 get_data_link dic_= ', dic)
    log_debug("3033-167 core views error 335")
    return JsonResponse(dic)

def get_data_json_link(request):
    dic_ = request.POST["dic"]
    dic_ = eval(dic_)
    # print("-2"*10)
    # print(dic_)
    # print("-2"*10)
    app_ = dic_["app"]
    model_ = dic_["model_name"]
    record_id_ = dic_["record_id"]
    model = apps.get_model(app_label=app_, model_name=model_)
    row = model.objects.get(id=record_id_)
    # print(row.friends)
    data = eval("row."+model_)
    dic = {'status': 'ok', "dic": data}
    return JsonResponse(dic)

def upload_file(request):
    print("9005", "\n", "-"*30)
    log_debug("upload_file 0000")
    upload_file_ = request.FILES['drive_file']
    ret = {}
    if upload_file_:
        filename = request.POST['filename']
        sheet_name_ = request.POST['sheet_name']
        # print("9005-1", filename, "\n", "-"*30)
        log_debug("upload_file 1111: "+filename)
        app_ = request.POST['app']
        # print("9005-2", app_, "\n", "-"*30)
        log_debug("upload_file 2222: "+app_)
        obj_name_ = request.POST['obj_name']
        function_name_ = request.POST['function_name']
        topic_id_ = request.POST['topic_id']
        folder_type_ = request.POST['folder_type']
        if folder_type_ == "media":
            print("media")
            data_dir = settings.MEDIA_ROOT + '/' + app_ + '/' + topic_id_
            log_debug("upload_file 3333: "+data_dir)

            upload_file_ = request.FILES['drive_file']
            os.makedirs(data_dir, exist_ok=True)
            filename = request.POST['filename']
            file_path = os.path.join(data_dir, filename)

            log_debug("upload_file 4444: "+file_path)

            model_name = request.POST['model_name']
            log_debug("upload_file 4444-0: ="+model_name+"=")
            log_debug("upload_file 4444-1: ")
            if model_name !="":
                record_id = request.POST['record_id']
                model_field_name = request.POST['model_field_name']
                model = apps.get_model(app_label=app_, model_name=model_name)
                obj = model.objects.get(id=record_id)
                s = "obj."+model_field_name+"='"+filename+"'"
                exec(s)
                obj.save()
                # print(file_path)
            log_debug("upload_file 5555:")
            with open(file_path, 'wb+') as destination:
                for c in upload_file_.chunks():
                    destination.write(c)
            ret['status'] = "ok"
            log_debug("upload_file 7777: OK")

            return HttpResponse(json.dumps(ret))
        # print("9005-16", folder_type_, "\n", "-"*30)
        #
        try:
            cube_dic = {}
            dimensions_ = request.POST['dimensions']
            fields_ = request.POST['fields']
            fact_model_field_ = request.POST['fact_model_field']
            # print("9005-161", fact_model_field_, "\n", "-"*30)
            #
            dimensions_s = dimensions_.split(",")
            # time_dim,country_dim,measure_dim
            # year,country_name,measure_name
            # WorldBankFact,amount
            fields_s = fields_.split(",")
            # print(fields_)
            # print(fact_model_field_)
            fact_model_field_s = fact_model_field_.split(",")
            cube_dic = {"dimensions": {}, "fact": {"model": fact_model_field_s[0], "field_name": fact_model_field_s[1]}}
            for j in range(len(dimensions_s)):
                f = fields_s[j]
                d = dimensions_s[j]
                dm = d.replace("_", "")
                cube_dic["dimensions"][d] = {"model": dm, "field_name": f}

            # print("90876 upload file: ")
            # print(cube_dic)
            # cube_dic = {"dimensions": {"time_dim": {"model": "TimeDim", "field_name": "year"},
            #                            "country_dim": {"model": "CountryDim", "field_name": "country_name"},
            #                            "measure_dim": {"model": "MeasureDim", "field_name": "measure_name"}},
            #             "fact": {"model": "WorldBankFact", "field_name": "amount"}
            #             }
            # print("90876-1 upload file: ")
            # print(cube_dic)
        except Exception as ex:
            pass

        add_dic = {"obj": obj_name_, "app": app_, "fun": function_name_,
                   "params": {"request": request, "folder_type": folder_type_, "sheet_name": sheet_name_, "app": app_, "cube_dic": cube_dic},
                   "obj_param": {"topic_id": topic_id_, "app": app_}}
        try:
            add_dic["obj_param"]["country_model"] = cube_dic["dimensions"]["country_dim"]["model"]
        except Exception as ex:
            pass
        try:
            add_dic["obj_param"]["measure_model"] = cube_dic["dimensions"]["measure_dim"]["model"]
        except Exception as ex:
            pass

        # print("9010")
        # print(add_dic)
        # print("9010")
        activate_obj_function(request, add_dic)
        # print("9011")
        ret['status'] = "ok"
    else:
        ret['status'] = "ko"

    return HttpResponse(json.dumps(ret))

def upload_file(request):
    # print("9005", "\n", "-"*30)
    log_debug("upload_file 0000")
    upload_file_ = request.FILES['drive_file']
    ret = {}
    if upload_file_:
        filename = request.POST['filename']
        sheet_name_ = request.POST['sheet_name']
        # print("9005-1", filename, "\n", "-"*30)
        log_debug("upload_file 1111: "+filename)
        app_ = request.POST['app']
        # print("9005-2", app_, "\n", "-"*30)
        log_debug("upload_file 2222: "+app_)
        obj_name_ = request.POST['obj_name']
        function_name_ = request.POST['function_name']
        topic_id_ = request.POST['topic_id']
        user_id_ = request.POST['user_id']
        folder_type_ = request.POST['folder_type']
        entity_name_ = request.POST['entity_name']
        if folder_type_ == "media":
            # print("media")
            data_dir = settings.MEDIA_ROOT + '/' + app_ + '/' + topic_id_
            log_debug("upload_file 3333: "+data_dir)

            upload_file_ = request.FILES['drive_file']
            os.makedirs(data_dir, exist_ok=True)
            filename = request.POST['filename']
            file_path = os.path.join(data_dir, filename)

            log_debug("upload_file 4444: "+file_path)

            model_name = request.POST['model_name']
            log_debug("upload_file 4444-0: ="+model_name+"=")
            log_debug("upload_file 4444-1: ")
            if model_name !="":
                record_id = request.POST['record_id']
                model_field_name = request.POST['model_field_name']
                model = apps.get_model(app_label=app_, model_name=model_name)
                obj = model.objects.get(id=record_id)
                s = "obj."+model_field_name+"='"+filename+"'"
                exec(s)
                obj.save()
                # print(file_path)
            log_debug("upload_file 5555:")
            with open(file_path, 'wb+') as destination:
                for c in upload_file_.chunks():
                    destination.write(c)
            ret['status'] = "ok"
            log_debug("upload_file 7777: OK")

            return HttpResponse(json.dumps(ret))
        #
        try:
            cube_dic = {}
            dimensions_ = request.POST['dimensions']
            fields_ = request.POST['fields']
            fact_model_field_ = request.POST['fact_model_field']
            # print("9005-161", fact_model_field_, "\n", "-"*30)
            #
            dimensions_s = dimensions_.split(",")
            # time_dim,country_dim,measure_dim
            # year,country_name,measure_name
            # WorldBankFact,amount
            fields_s = fields_.split(",")
            # print(fields_)
            # print(fact_model_field_)
            fact_model_field_s = fact_model_field_.split(",")
            cube_dic = {"dimensions": {}, "fact": {"model": fact_model_field_s[0], "field_name": fact_model_field_s[1]}}
            for j in range(len(dimensions_s)):
                f = fields_s[j]
                d = dimensions_s[j]
                dm = d.replace("_", "")
                cube_dic["dimensions"][d] = {"model": dm, "field_name": f}

            # print("90876 upload file: ")
            # print(cube_dic)
            # cube_dic = {"dimensions": {"time_dim": {"model": "TimeDim", "field_name": "year"},
            #                            "country_dim": {"model": "CountryDim", "field_name": "country_name"},
            #                            "measure_dim": {"model": "MeasureDim", "field_name": "measure_name"}},
            #             "fact": {"model": "WorldBankFact", "field_name": "amount"}
            #             }
            # print("90876-1 upload file: ")
            # print(cube_dic)
        except Exception as ex:
            pass

        add_dic = {"obj": obj_name_, "app": app_, "fun": function_name_,
                   "params": {"request": request, "folder_type": folder_type_, "sheet_name": sheet_name_,
                              "user_id": user_id_, "app": app_, "cube_dic": cube_dic},
                   "obj_param": {"topic_id": topic_id_, "app": app_, "entity_name": entity_name_}}

        try:
            add_dic["obj_param"]["entity_model"] = cube_dic["dimensions"][entity_name_+"_dim"]["model"]
        except Exception as ex:
            pass
        try:
            add_dic["obj_param"]["measure_model"] = cube_dic["dimensions"]["measure_dim"]["model"]
        except Exception as ex:
            pass

        # print("9010")
        # print(add_dic)
        # print("9010")
        return activate_obj_function(request, add_dic)
        # print(ret)
        # print("9011")
        # ret['status'] = "ok"
    else:
        ret['status'] = "ko"

    # print("ret", ret)
    return HttpResponse(json.dumps(ret))

def logmein(request):
    try:
        dic_ = request.POST["dic"]
        dic_ = eval(dic_)
    except Exception as ex:
        print("Error 200-21: "+str(ex))

    # print('90155-100 logmein\n','-'*50, "\n", dic_, "\n", '-'*50)
    username= dic_['username']
    password = dic_['pwd']
    app_ = dic_['app']
    try:
        user = authenticate(username=username, password=password)
    except Exception as ex:
        print(ex)

    if user is not None:
       if user.is_active:
           login(request, user)
           return JsonResponse({"status":"ok"})
       else:
           return JsonResponse({"status":"inactive"})
    else:
       return JsonResponse({"status":"bad"})

def delete_record_from_table(request):
    ret = {"status":"ko"}
    try:
        dic_ = request.POST["dic"]
        dic_ = eval(dic_)
        # print(dic_)
        app = dic_["app"]
        model = dic_["model"]
        record_id = dic_["record_id"]
        model_ = apps.get_model(app_label=app, model_name=model)
        obj = model_.objects.get(id=record_id)
        obj.delete()
        ret = {"status":"ok"}
    except Exception as ex:
        print("Error 200-21: "+str(ex))
    return JsonResponse(ret)

def general_data(request):
    dic_ = request.POST["dic"]
    dic_ = eval(dic_)
    # try:
    #     print('9050-1 core views general_data dic_= \n', dic_, '\n', "-"*50)
    # except Exception as ex:
    #     pass
    general_data_model = apps.get_model(app_label="core", model_name="generaldata")
    action_= dic_['action']
    app_= dic_['app']
    group_ = dic_['group']
    data_name_ = dic_['data_name']
    result = {"status": "ok", "json":{}}
    if action_ == "set":
        data_json_ = dic_['data_json']
        obj, is_created = general_data_model.objects.get_or_create(app=app_, group=group_, data_name = data_name_)
        obj.data_json = data_json_
        obj.save()
    else:
        obj = general_data_model.objects.get(app=app_, group=group_, data_name = data_name_)
        result["json"] = obj.data_json
    return JsonResponse(result)

