import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.apps import apps
from django.db.models.fields.related import ForeignKey
from .models import DataAdvancedTabs, DataAdvancedTabsManager, AdjectivesValues, DataAdvancedApps


class AdvancedTabs(object):
    def __init__(self, manager_name, app):
        self.manager_name = manager_name
        self.app = app

    def delete_tab(self, params):
        try:
            # print('-2'*10)
            # print(params)
            # print('-2'*10)
            tab_name_ = params["tab_name"]
            tab_to_delete = DataAdvancedTabs.objects.get(manager__at_name=self.manager_name, tab_name=tab_name_)
            tab_id_ = tab_to_delete.id
            tab_to_delete.delete()
            result = {'tab_id': tab_id_}
        except Exception as ex:
            result = {'tab_id': "-1"}
        return result

    def delete_record(self, params):
        try:
            # print(params)
            model = apps.get_model(app_label=params["app"], model_name=params["model"])
            obj = model.objects.get(id=params["id"])
            obj.delete()
            result = {'id': params["id"]}
        except Exception as ex:
            result = {'tab_id': "-1"}
        return result

    def add_tab(self, params):
        try:
            # print('add_tab 908654-1')
            # print(params)
            # print('-1'*10)

            tab_name_ = params["tab_name"]
            manager_, n_ = DataAdvancedTabsManager.objects.get_or_create(at_name=self.manager_name)
            if n_:
                manager_.manager_content = {"last_obj_number": 0}
                manager_.save()
            t, n_ = DataAdvancedTabs.objects.get_or_create(manager=manager_, tab_name=tab_name_)
            try:
                content_ = {"properties": {"tab_name": params["tab_name"], 'tab_order': 1,
                                           "tab_title": params["tab_name"],
                            "tab_type": "empty"}}
                # print(content_)
                t.tab_content = content_
                t.save()
            except Exception as ex:
                print(ex)
            result = {t.id: t.tab_content}
        except Exception as ex:
            result = {'error': "-1"}
        return result

    def get_tabs_from_table(self, params):
        # print('get_tabs_from_table 90555-1')
        # print(params)
        # print(self.app)
        # print(self.manager_name)
        # print('-2'*10)
        result = {}
        try:
            manager_ = DataAdvancedTabsManager.objects.get(at_name=self.manager_name)
            result["manager"] = manager_.manager_content
            tabs = DataAdvancedTabs.objects.filter(manager=manager_).all()
            result_ = []
            for t in tabs:
                # print(t.id)
                # print('t.tab_content')
                content_ = t.tab_content
                content_["properties"]["tab_order"] = t.order
                content_["tab_order"] = t.order
                t.tab_content = content_
                t.save()
                result_.append((t.id, t.tab_content))
            result["tabs"] = result_
        except Exception as ex:
            pass
            # result = {"error": "-1"}
        try:
            app_, c = DataAdvancedApps.objects.get_or_create(app_name=self.app)
            # print(app_)
            # print(app_.app_content)
            result["app_content"] = app_.app_content
        except Exception as ex:
            result = {"error": "-1"}
        # print(result)
        return result

    def save_content(self, params):
        try:
            # print('save_content 906543-1')
            # print('='*50)
            # print('params')
            # print(params)
            # print('atm_content')
            # print(params["atm_content"])
            # print('='*20)
            # print('9075-1: AdvancedTabs tab_content\n', "-"*50, params["tab_content"], "\n", "-"*50)
            # print('tab_name')
            # print(params["tab_name"])
            # print(params["app_content"])
            # print('='*50)
            try:
                atm = DataAdvancedTabsManager.objects.get(at_name=self.manager_name)
                atm.manager_content = params["atm_content"]
                atm.save()
                app = DataAdvancedApps.objects.get(app_name=self.app)
                app.app_content = params["app_content"]
                app.save()
                tab = DataAdvancedTabs.objects.get(manager=atm, id=params["tab_id"])
                tab.tab_content = params["tab_content"]
                tab.tab_name = params["tab_name"]
                tab.order = params["tab_order"]
                tab.save()
                # print("saved")
            except Exception as ex:
                print(ex)
            # print(tab)
            # print('='*50)
            result = {"saved": "ok"}

        except Exception as ex:
            result = {"error": "-1"}
        return result

    def get_adjective(self, params):
        # print('='*50, '\ncore AdvancedTabes\nparams\n', params, "\n", '='*50)

        app_ = params['app']
        model_name_ = params['model_name']
        field_value_ = params['field_value']
        index_="id"
        try:
            index_ = params['index']
        except Exception as ex:
            pass
        manager_name = params['manager_name']
        manager_fun = params['manager_fun']
        manager_fun_field = params['manager_fun_field']
        model = apps.get_model(app_label=app_, model_name=model_name_)

        # model.adjectives.adjectives(adjective_title='accounting_report')
        s = "model." + manager_name + "." + manager_fun + "("+manager_fun_field+"='"+field_value_+"')"
        # print(s)
        data = eval(s)
        # print(data)
        result = []
        for q in data:
            # print(q.order, q.id)
            if index_ == "order":
                result.append((q.order, q.value))
            else:
                result.append((q.id, q.value))
        # print('result', result, 'result')
        return result

    def get_list_from_model(self, params):
        # print('-'*50, '90345 params', "\n", params, "\n", '-'*50)
        app_ = params['app']
        model_name_ = params['model_name']
        field_name_ = params['field_name']
        model = apps.get_model(app_label=app_, model_name=model_name_)

        # print(model._meta.get_fields(include_hidden=True))
        # for field in model._meta.get_fields(include_hidden=True):
        #     print(field.name)

        data_filter_field_ = params['data_filter_field']
        data_filter_field_value_ = params['data_filter_field_value']
        data_filter_field_ft_ = ""
        try:
            data_filter_field_ft_ = params['data_filter_field_ft']
        except Exception as ex:
            pass
            # print("90876-23: "+str(ex))
        squ = ''
        try:
            if model.model_field_exists(app_+'_web') and isinstance(model._meta.get_field(app_+'_web'), ForeignKey):
                company_obj_id_ = params['company_obj_id']
                parent_model = apps.get_model(app_label=app_, model_name=app_ + "web")
                company_obj = parent_model.objects.get(id=company_obj_id_)
                squ += 'model.objects.filter('+app_ + '_web=company_obj '
        except Exception as ex:
            print("9025 "+str(ex))

        # print('901544 params \nsqu= '+squ)
        if data_filter_field_value_ != "":
            # print('901555 params')
            if squ == '':
                squ += 'model.objects.filter('
            else:
                squ += ','
            # print("9013 \n"+squ)

            if data_filter_field_ft_ and data_filter_field_ft_ == "Yes":
                try:
                    squ_ = squ + data_filter_field_ + '__id=' + data_filter_field_value_ + ').all()'
                    # print('squ-2220:\n', squ_)
                    data = eval(squ_)
                    squ = squ_
                except Exception as ex:
                    squ_ = squ + data_filter_field_ + '=' + data_filter_field_value_ + ').all()'
                    # print('squ-2222:\n', squ_)
                    data = eval(squ_)
                    squ = squ_
            else:
                try:
                    squ += data_filter_field_+'__icontains="'+data_filter_field_value_+'").all()'
                    print("9014 \n"+squ)
                    data = eval(squ)
                except Exception as ex:
                    # print("9015-1 \n"+squ)
                    squ += data_filter_field_+'__icontains='+data_filter_field_value_+').all()'
                    data = eval(squ)
        else:
            if squ == '':
                squ += 'model.objects.filter('
            squ += ').all()'
            # print("9064 \n"+squ)
            data = eval(squ)
            # data = model.objects.all()
        # print(data)
        # df = pd.DataFrame(data.values())
        # print(df)
        # print('901544-1 params \nsqu= \n'+squ)

        p_key_field_name = model._meta.pk.name
        if  p_key_field_name != "id":
            for field in model._meta.get_fields(include_hidden=True):
                if field.name == p_key_field_name+"_id":
                    p_key_field_name =  p_key_field_name+"_id"
                    break
        result = []
        for q in data:
            s = 'result.append((q.'+ p_key_field_name+', q.'+field_name_+'))'
            # print(s)
            eval(s)

        # print('result')
        # print(result)
        # print('result')
        return result

    def truncate_model(self, params):
        # print('9015 params')
        # print(params)
        app_ = params['app']
        model_name_ = params['model_name']
        try:
            model = apps.get_model(app_label=app_, model_name=model_name_)
            model.truncate()
        except Exception as ex:
            print("9025 "+str(ex))

        result = "Data truncated"
        return result

