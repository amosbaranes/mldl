// -- AdvancedTabsManager --
//function AdvancedTabsManager(my_name, my_app, body_,
//activate_function_link, update_file_link, activate_obj_function_link,upload_file_link,logmein_link,get_data_link,
//get_data_json_link,company_obj_id, is_show_btns=true, user_id=0)
//{
//}

// -- AdvancedTabsManager --
function AdvancedTabsManager(dic=null)
{
// "Tab":{"title":"Tab", "obj_type":"none",
//                        "sub_buttons": {"NewFunction":{"title":"new func", "width":10},
//                                        "DeleteFunction":{"title":"del func", "width":10}}},

 this.buttons =
 {"Tab":{"title":"Tab", "obj_type":"none",
                        "sub_buttons": {}},
                 "TabContent":{"title":"Tab Content", "obj_type":"none",
                        "sub_buttons": {"content":{"title":"Content", "width":10, "setting":[], "attributes":[], "functions":[]}}},
                 "Component":{"title":"Component", "obj_type":"acObj",
                        "sub_buttons": {"Button":{"title":"Button", "width":5,
                                                  "setting": {"color":[],"background-color":[], "font-weight":["normal","lighter","bold","900"]},
                                                  "attributes":{},
                                                  "functions":["onmouseover", "onmouseout"]},
                                        "Span":{"title":"Span", "width":5,
                                                "setting": {"color":[], "background-color":[],
                                                            "font-weight":["normal","lighter","bold","900"],
                                                            "text-align":["", "left", "center", "right"]},
                                                "attributes":{},
                                                "functions":["onmouseover", "onmouseout"]},
                                        "Input":{"title":"Input", "width":5,
                                                 "setting": {"color":[], "background-color":[],
                                                             "text-align":["", "left", "center", "right"], "span_width":[]},
                                                 "attributes":{"field":[],
                                                               "type":["text","button","checkbox","color","date",
                                                                       "datetime-local","email","file","hidden","image",
                                                                       "month","number","password","radio","range","reset",
                                                                       "search","submit","tel","time","url","week"],
                                                               "foreign_table":[],
                                                               "dependent":[],
                                                               "min":[],"max":[],"step":[]},
                                                 "functions":["onchange","oninput", "onkeyup", "onkeydown", "onload"]},
                                        "Select":{"title":"Select", "width":5,
                                                  "setting": {"options":[], "global_adjective":[], "app_adjective":[],
                                                              "index":["", "id", "order"],
                                                              "data_app":[], "data_model":[],"data_field":[],
                                                              "data_filter_field":[], "data_filter_field_value":[],
                                                              "data_filter_field_ft":["","Yes"],
                                                              "multiple":["regular", "multiple"],
                                                              "default_value":[], "call_on_change":["", true, false]},
                                                  "attributes":{"field":[], "size":[], "foreign_table":[],
                                                                "dependent":[]},
                                                  "functions":["onchange"]},
                                        "Table":{"title":"Table", "width":5, "setting": {}, "attributes":{}, "functions":["onchange"]},
                                        "Textarea":{"title":"textarea", "width":7,
                                                    "setting": {"overflow":[],"color":[],"background-color":[]},
                                                    "attributes":{"field":[], "rows":[], "cols":[]}, "functions":[]},
                                        "RichText":{"title":"RichText", "width":7,
                                                    "setting": {"height":[], "width":[]},
                                                    "attributes":{"field":[]},
                                                    "functions":["onchange"]},
                                        "Object":{"title":"object", "width":7,
                                                    "setting": {"color":[],"background-color":[]},
                                                    "attributes":{"data":[], "width":[], "height":[]},
                                                    "functions":[]},
                                        "Canvas":{"title":"canvas", "width":7,
                                                    "setting": {"overflow":[],"color":[],"background-color":[]},
                                                    "attributes":{"width":[], "height":[]}, "functions":[]},
                                        "DIV":{"title":"div", "width":3,
                                               "setting": {"overflow":["visible", "clip", "auto", "hidden"],
                                                           "width":[], "height":[], "color":[],"background-color":[],
                                                           "text-align":["", "left", "center", "right"]
                                               },
                                               "attributes":{},
                                               "functions":[]},
                                        "A":{"title":"a", "width":3,
                                             "setting": {"color":[], "background-color":[]},
                                             "attributes":{"href":[], "target":["", "blank_"], "download":["", "download"]},
                                             "functions":[]},
                                        "H":{"title":"h", "width":3,
                                             "setting": {"font-size":[], "font-family":["", "Arial, sans-serif",
                                                         "Verdana, sans-serif","Tahoma, sans-serif","'Trebuchet MS', sans-serif",
                                                         "'Times New Roman', serif","Georgia, serif", "Garamond, serif",
                                                         "'Courier New', monospace", "'Brush Script MT', cursive"
                                                         ]},
                                             "attributes":{"color":[], "background-color":[]},
                                             "functions":[]},
                                        "H1":{"title":"h1", "width":3,
                                              "setting": {"color":[], "background-color":[]},
                                              "attributes":{}, "functions":[]},
                                        "H2":{"title":"h2", "width":3,
                                              "setting": {"color":[], "background-color":[]},
                                              "attributes":{}, "functions":[]},
                                        "H3":{"title":"h3", "width":3,
                                              "setting": {"color":[], "background-color":[]},
                                              "attributes":{}, "functions":[]}}},
                 "TabNavLink":{"title":"Tab Nav Link", "obj_type":"none",
                        "sub_buttons": {"nav":{"title":"Navigator", "width":10},
                                        "item":{"title":"Item", "width":10}}},
                 "PopWin":{"title":"Pop Win", "obj_type":"PopWin",
                        "sub_buttons": {"NewPopWin":{"title":"New Pop Win", "width":10}
//                        ,
//                                        "DeletePopWin":{"title":"Del Pop Win", "width":10}
                                        }
                                        },
                 "Plugin":{"title":"Plugins", "obj_type":"acPlugin",
                        "sub_buttons": {"Container":{"title":"Container", "width":7,
                                                     "setting": {"height":[], "width":[], "color":[],
                                                                 "background-color":[], "z_index":[-1,0,1,2,3,4,5]},
                                                     "attributes":{"table":[], "parent_table":[], "content_type":[],
                                                                   "border_style":["none","solid","Dotted","dashed",
                                                                   "double","groove","ridge","inset","outset",
                                                                   "hidden"], "border_width":[],
                                                                   "border_color":[], "border_radius":[]},
                                                     "functions":["onclick", "onchange", "onmouseover", "onmouseout"]
                                                    },
                                        "Radio": {"title":"Radio", "width":5,
                                                  "setting": {"setup_dictionary":[]},
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "Test":{"title":"T", "width":3,
                                                  "setting": {"setup_dictionary":[], "width":[], "height":[],
                                                              "table":[], "field":[], "value_field":[],
                                                              "recording_tests_number":[]},
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "Variable":{"title":"V", "width":3,
                                                  "setting": {"setup_dictionary":[], "width":[], "height":[],
                                                              "table":[], "field":[], "value_field":[],
                                                              "recording_variables_number":[]},
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "Group":{"title":"G", "width":3,
                                                  "setting": {"setup_dictionary":[], "entity":[], "width":[],
                                                  //"height":[],
                                                              "color":[], "background_color":[],
                                                              "border_style":["none","solid","Dotted","dashed",
                                                                   "double","groove","ridge","inset","outset","hidden"],
                                                              "border_width":[],"border_color":[], "border_radius":[],
                                                              "table":[], "field":[], "value_field":[],
                                                              "recording_tests_number":[],
                                                              "present":["","groups", "detail"]},
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "RecordingTests":{"title":"R", "width":3,
                                                  "setting": {"entity":[],"width":[],"userid_width":[],"name_width":[],
                                                  "height":[],"table":[],"vertical_field":[],"horizontal_field":[],
                                                  "field":[]},"attributes":{"table_class":["","basic","payment","data_entry"]},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "Compliance":{"title":"C", "width":3,
                                                  "setting": {"width":[],
                                                  "week_table":[], "day_table":[]},
                                                  "attributes":{"table_class":["","basic","payment","data_entry"]},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "Image":{"title":"Img", "width":4,
                                                  "setting": {"height":[], "src":[], "width":[], "border_radius":[],
                                                  "border_width":[], "border_color":[]
                                                  },
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "on_loaded", "onmouseover", "onmouseout"]},
                                        "Video":{"title":"Vid", "width":4,
                                        "setting": {"height":[], "src":[], "width":[]},
                                        "attributes":{},
                                        "functions":["onclick", "onchange", "on_loaded", "onmouseover", "onmouseout"]},
                                        "Activity": {"title":"Activity", "width":6,
                                                  "setting": {"table_class":["","basic", "payment"], "parent_table":[], "table":[],
                                                              "number_of_radios":[]},
                                                  "attributes":{},
                                                  "functions":["onclick", "onchange", "onmouseover", "onmouseout"]},
                                        "SearchTable":{"title":"Search Table", "width":10,
                                                       "setting": {"is_new_button":["","Yes","No"],
                                                       "is_del_button":["","Yes","No"], "call_get_data":["", true, false]},
                                                       "attributes":{"number_of_rows":[], "table_class":["","basic",
                                                       "payment"], "height":[]},
                                                       "functions":["onclick", "onchange", "onmouseover", "onmouseout", "on_new_record", "before_get_data", "on_get_data"],
                                                       "field_setting":["field_title","field_name","field_width","field_align","foreign_table","foreign_table_field","filter"]},
                                        "MSearchTable":{"title":"MSearchTable", "width":10,
                                                       "setting": {"is_new_button":["","Yes","No"], "is_del_button":["","Yes","No"]},
                                                       "attributes":{"number_of_rows":[], "table_class":["","basic", "payment"], "height":[]},
                                                       "functions":["onchange", "before_get_data"],
                                                       "field_setting":["field_title","field_name","field_width","field_align","foreign_table"]},
                                        "GeneralLedger":{"title":"GenLedger", "width":8,
                                                         "setting": {},
                                                         "attributes":{"table_class":["","basic", "payment"], "table":[]},
                                                         "functions":["onchange"],
                                                         "field_setting":["field_title","field_name","field_width","field_align"]},
                                        "DETable":{"title":"DaEntryTable", "width":10,
                                                       "setting": {"is_new_button":["","Yes","No"], "is_del_button":["","Yes","No"]},
                                                       "attributes":{"table":[], "table_class":["","basic", "payment", "data_entry"], "height":[]},
                                                       "functions":["onchange", "on_new_record"],
                                                       "field_setting":{"field_title":[],"field":[],
                                                                        "field_type":["text","button","checkbox","color",
                                                                                      "date","datetime-local","email",
                                                                                      "file","hidden","image","month",
                                                                                      "number","password","radio",
                                                                                      "range","reset","search","submit",
                                                                                      "tel","time","url","week","select"]
                                                                       ,"field_width":[],"field_align":[],
                                                                       "foreign_table":[],"filter":[]}},
                                        "Chart":{"title":"Chart", "width":5,
                                                 "setting": {
                                                   "remove_buttons":["", "true", "false"],
                                                   "remove_markers":["", "true", "false"],
                                                   "remove_hovermode":["", "true", "false"]},
                                                   "attributes":{"height":[], "interval":[]},
                                                   "functions":[]},
                                        "Candle":{"title":"Candle", "width":6,
                                                 "setting": {"border_width":[],"border_color":[],
                                                             "scale_type":["linear", "logarithmic"],"height":[],"width":[]},
                                                 "attributes":{},
                                                 "functions":[]},
                                        "Candle":{"title":"Candle", "width":6,
                                                 "setting": {"border_width":[],"border_color":[],
                                                             "scale_type":["linear", "logarithmic"],"height":[],"width":[]},
                                                 "attributes":{},
                                                 "functions":[]},
                                        "UploadFile":{"title":"UploadFile", "width":8,
                                               "setting": {"height":[],"width":[],"obj_name":[],"function_name":[],
                                                           "topic_id":[], "folder_type":["", "excel", "media", "mediafun", "other"],
                                                            "dimensions":[], "fields":[], "fact_model_field": [],
                                                            "entity_name":[], "sheet_name":[]},
                                               "attributes":{"field":[]},
                                               "functions":["onchange", "on_loaded"]},
                                        "Heatmap":{"title":"Heatmap", "width":7,
                                                   "setting": {"width":[], "height":[]},
                                                   "attributes":{},
                                                   "functions":[]}}},
                 "Dashboard":{"title":"Dash Board", "obj_type":"acDashboard",
                        "sub_buttons":{"Basic":{"title":"Basic", "width":10,
                                                "setting": {"app":[], "table":[], "setup_dictionary":[], "entity":[],
                                                            "type":["", "individual", "unit"], "synchronize_with":[],
                                                            "height":[], "width":[], "kpis":[], "kpis_sub_titles":[],
                                                            "kpis_images":[], "kpis_progress_co":[], "skills":[],
                                                            "border_style":["none","solid","Dotted","dashed",
                                                                   "double","groove","ridge","inset","outset",
                                                                   "hidden"],
                                                             "border_width":[],"border_color":[], "border_radius":[],
                                                             "board1_color":[],"board2_color":[],"board3_color":[],"board4_color":[],
                                                             "company_logo":[], "battalion_logo":[]
                                                },
                                                "attributes":{"background-color":[], "color":[], "table_class":["","basic", "payment"]},
                                                "functions":["on_amount_paint", "on_receive_data", "on_entity_click",
                                                             "onclick", "onmouseover", "onmouseout"]
                                               }
                        }
                 },
                 "Report":{"title":"Reports", "obj_type":"acReport",
                        "sub_buttons": {"Pivot":{"title":"Pivot Table", "width":10,
                                                 "setting": {"app":[], "table":[],
                                                 "vertical_field":[],"vertical_title":[],
                                                 "horizontal_field":[],"horizontal_title":[],
                                                 "value_field":[], "order_by":[],"height":[]},
                                                 "attributes":{"table_class":["","basic", "payment"]},
                                                 "functions":["on_amount_paint", "on_receive_data", "onclick", "onmouseover", "onmouseout"]
                                                },
                                       "FSPivot":{"title":"Financial Statements Pivot", "width":20,
                                                   "attributes":{"table_class":["","basic", "payment"]},
                                                   "setting": {"app":[],"table":[],"vertical_field":[],
                                                   "horizontal_field":[],"horizontal_title":[],"value_field":[], "order_by":[],
                                                   "statement_type":["BalanceSheet","Income","CashFlow","ChangInEquity"]},
                                                   "functions":["on_amount_paint", "on_receive_data", "onclick", "onmouseover", "onmouseout"]
                                                  },
                                       "Grades":{"title":"Grades", "width":20,
                                                   "attributes":{"table_class":["","basic", "payment"]},
                                                   "setting": {"app":[],"table":[],"vertical_field":[],
                                                   "image":[],
                                                   "horizontal_field":[],"horizontal_title":[],"value_field":[], "order_by":[]},
                                                   "functions":["on_amount_paint", "on_receive_data", "onclick", "onmouseover", "onmouseout"]
                                                  },
                                       "DashBoard":{"title":"Dash Board", "width":20,
                                                   "attributes":{"table_class":["","basic", "payment"]},
                                                   "setting": {"app":[],"table":[],
                                                   },
                                                   "functions":["on_amount_paint", "on_receive_data", "onclick", "onmouseover", "onmouseout"]
                                                  }
                                       }
                           },
                 "Admin":{"title":"Admins", "obj_type":"acAdmin",
                        "sub_buttons": {"ToDo":{"title":"To Do", "width":10,
                                                "setting": {"width":[], "table":[]},
                                                "attributes":{"table_class":["","basic","payment","data_entry"]},
                                                "functions":["onclick", "onchange", "onmouseover", "onmouseout"]
                                                }
                                       }
                         },
                 "AppObjs":{"title":"App objects", "obj_type":"none",
                        "sub_buttons": {}
                           }
                }

 this.tab = {"functions":[],
             "settings_list":{"button_color":[],"button_bg_color":[],
                              "font-weight":["normal","lighter","bold","900"],
                              "width":[],"height":[],
                              "border_radius":[],
                              "tab_name":[],
                              "tab_type":["simple", "navmulti", "navone"],
                              "tab_order":[],"tab_title":[],"link_number":[]},
             "attributes_list":{}
            }

 this.tab_nav_links = {"functions":["onclick", "onmouseover", "onmouseout"],
                       "settings_list":{"width":[], "add_title":[], "remove_title":[], "is_show_btns":["", "true", "false"],
                                        "left":[], "top":[], "border_radius":[],"button_color":[],"button_bg_color":[],
                                        "font-weight":[], "height":[], "button_align":["", "horizontal", "vertical"],
                                        "obj_number":[], "background_color":[]}, "attributes_list":{}
                       }

 this.nav_link = {"functions":["onclick", "onmouseover", "onmouseout"],
                  "settings_list":{"link_number":[],"link":[],"title":[]},
                  "attributes_list":{}
                  }

 this.tab_content = {"functions_list":["onclick", "onchange", "onmouseover", "onmouseout"],
                     "settings_list":{"width":[], "color":[], "background-color":[], "company_obj_id":[]},
                     "attributes_list":{"table":[], "parent_table":[], "link_number":[], "content_type":[]}
                     }
 this.pop_win = {"functions_list":["onclick","onmouseover","onmouseout","__init___","__get_panel_structure___","__set_panel___"],
                 "settings_list":{"width":[], "height":[], "color":[], "background_color":[]},
                 "attributes_list":{"name":[],"top":[],"right":[],"title":[], "table":[], "link_number":[],"tab_id":[],
                                    "is_panel":["true", "false"],"title_color":[], "title_background_color":[],
                                    "content_type":[]}}
 if(dic!=null){this.create_atm(dic)}
}

AdvancedTabsManager.prototype.create_atm = function(dic)
{
 //alert("create_atm \n"+JSON.stringify(dic))
 try{this.add_delete_editor.outerHTML="";this.titles.outerHTML="";this.container.outerHTML="";} catch(er){}
 this.dic=dic;
 this.my_name=dic["my_name"];
 this.my_app=dic["my_app"];
 this.elm_body=dic["body_"];
 this.logmein_link_=dic["logmein_link"];
 this.delete_record_from_table_link_=dic["delete_record_from_table_link"];
 this.activate_function_link_=dic["activate_function_link"];
 this.update_field_link_=dic["update_field_link"];
 this.activate_obj_function_link_=dic["activate_obj_function_link"];
 this.upload_file_link_ = dic["upload_file_link"]
 this.get_data_link_=dic["get_data_link"];
 this.general_data_=dic["general_data"]
 this.get_data_json_link_=dic["get_data_json_link"];
 this.company_obj_id=dic["company_obj_id"];
 this.user_id=dic["user_id"];
 //--
 this.default_battalion=dic["default_battalion"];
 this.default_entity=dic["default_entity"];
 this.entity_name=dic["entity_name"];
 try{var s="this.default_"+this.entity_name+"="+this.default_entity;eval(s)} catch(er){}
 // for the ch application, default_entity is the same as default_branch
 //--

 //alert("atm: "+this.default_battalion)
 //--------------------------
 this.titles=null;this.container=null;this.content={"last_obj_number":0};this.temp_number=1;
 this.app_content={};this.app_objs={};this.tabs={};this.pop_wins={};
 //--------------------------
 this.init_create_containers();
 this.fun_to_run_by_timer=[];this.interval=100;this.init_timer();
 this.is_show_btns = dic["is_show_btns"]
 if(dic["is_show_btns"] == true){this.create_add_delete_editor();}
 this.setTabs();
 //--------------------------
 this.active_tab=null;this.editor=null;this.new_obj_to_create=null;
 atm = this;
 return this;
}

AdvancedTabsManager.prototype.init_create_containers = function()
{
   this.add_delete_editor = document.createElement("div");
   this.titles = document.createElement("div");
   this.titles.setAttribute("class", "maintab");
//   var titles_height=50
//   this.titles_padding = 10
//   this.titles_width=170;
//   var titles_background_color = "green"
//   this.titles_height=titles_height-2*this.titles_padding;
//   var style_="text-align:center;background-color:"+titles_background_color+";height:"+titles_height+"px;padding:5px;";
//   this.titles.setAttribute("style", style_)

   this.titles_height=0;
   this.container = document.createElement("div");
   this.container.setAttribute("style", "width: 100%;height: 100%;")

   this.panel = document.createElement("div");
   this.panel.setAttribute("style", "width: 100%;")
   this.elm_body.appendChild(this.add_delete_editor);
   this.elm_body.appendChild(this.panel);
   this.elm_body.appendChild(this.titles);
   this.elm_body.appendChild(this.container);
}

AdvancedTabsManager.prototype.init_timer = function()
{
    fun=function(this_obj){
      //alert("in this.fun\n"+JSON.stringify(this_obj.fun_to_run_by_timer))
      var date = new Date;
      this.date=date;
      this_obj.h=date.getHours();this_obj.m=date.getMinutes();this_obj.s=date.getSeconds();
      this_obj.d=date.getDay();this_obj.y=date.getFullYear();this_obj.day_of_month=date.getDate()
      this_obj.month_number=date.getMonth();

      this_obj.day_name=days_of_name[this_obj.d]

      //alert("1 "+this_obj.day_name)
       //alert(this_obj.h +":"+ this_obj.m +":"+ this_obj.s +":"+ this_obj.y)
      for(i in this_obj.fun_to_run_by_timer){
      //alert(i)
        var dic=this_obj.fun_to_run_by_timer[i];
        if(1*dic["run"]==1){
          //alert(i+" : "+dic["fun_name"])
          var obj = dic["obj_"];
          //alert(obj.outerHTML)
          //alert(dic["fun_ref"]);
          try{
            eval(dic["fun_ref"]);
          } catch(er){alert(er)}
          dic["run"]=(1*dic["interval"])/this_obj.interval
        //alert("this_obj.interval: "+this_obj.interval+" interval: "+dic["interval"]+" :in: "+dic["run"])
        } else {dic["run"]=1*dic["run"]-1}
        //alert(dic["interval"]+" : "+dic["run"])
      }
    }
    try{Interval=setInterval(fun, this.interval, this)} catch(er){alert(er)}
}

AdvancedTabsManager.prototype.set_fun_for_timer = function(fun_name, fun_ref, interval, obj=null)
{
  //alert(obj);
  var run_=1*interval/this.interval
  //alert(run_)
  dic_={"fun_name":fun_name, "fun_ref":fun_ref, "interval":interval, "run":run_, "obj_": obj}
  if(this.fun_to_run_by_timer.length>0)
  {
     for(i in this.fun_to_run_by_timer){
         var fun_name_=this.fun_to_run_by_timer[i]["fun_name"];
         if(fun_name_==fun_name){return;}
     }
  }
  this.fun_to_run_by_timer.push(dic_)
}

AdvancedTabsManager.prototype.create_add_delete_editor = function()
{
  this.add_btn=document.createElement("button");
  this.add_btn.innerHTML="Add Tab";
  this.add_btn.addEventListener("click", function(){
      var tab_name_ = prompt("Enter name for new tab:",'');if(tab_name_==''){alert("Please enter a tab name"); return;}
      tab_name_=tab_name_.toLowerCase();
      var dic_ = {"obj" : "AdvancedTabs", "atm": atm_.my_name, "app": atm_.my_app, "fun": "add_tab", "params": {"tab_name": tab_name_}}
      //alert(atm_.activate_function_link_)
             $.post(atm_.activate_function_link_,
                  {dic : JSON.stringify(dic_)},
                  function(dic){
                    //alert(JSON.stringify('dic'))
                    //alert(JSON.stringify(dic))
                    var result = dic["result"]
                    for (id_ in result){
                     result[id_]["properties"]["link_number"]=this.get_next_obj_number();
                     atm.tabs[id_] = new Tab(atm, data=result[id_], id=id_);
                    }
                    //alert(JSON.stringify(result))
                    atm.tabs[id_].btn.click();
                    try{atm.set_active_tab(atm.tabs[id_].btn)} catch(er){alert('er9037: '+ er)}
                    try{atm.save()} catch(er){alert("er900: "+er)}
                  }.bind(atm=atm_));
      }.bind(atm_=this, event))
  //--
  this.delete_btn = document.createElement("button");
  this.delete_btn.innerHTML = "Delete Tab"
  this.delete_btn.addEventListener("click", function(){
       var tab_name_ = prompt("Enter name of a tab to delete:" , '')
       if(tab_name_ == '') {alert("Please enter a tab name"); return;}
       var dic_ = {"obj" : "AdvancedTabs", "atm": atm_.my_name, "app": this.my_app, "fun": "delete_tab", "params": {"tab_name": tab_name_}}
       // alert(JSON.stringify(dic_))
              $.post(atm_.activate_function_link_,
              {
                dic : JSON.stringify(dic_)
              },
              function(dic){
                //alert(JSON.stringify(dic))
                var tab_id_ = dic["result"]["tab_id"];
                atm.delete_tab(tab_id_, tab_name_);
             }.bind(atm=atm_))
  }.bind(atm_=this, event))
  //--
  this.editor_btn=document.createElement("button");
  this.editor_btn.innerHTML = "Editor"
  this.editor_btn.parent=this;
  this.editor_btn.addEventListener("click", function(){
  if(atm_.editor == null){
    var editor = new PopWin(my_name_="editor",win_name_="editor",win_title_="Editor for Tab: ",user_id=1, atm=atm_)
    editor.__init__();
    editor.set_win_frame_style("20", "650", "1000", "15%", "20%", "white")
    editor.set_acWinStatEventListeners(editor);
    editor.set_acWinStat('block');
    atm_.editor=editor;
  }
    try{editor.set_tab(atm_.active_tab);} catch(er){
        //    var editor = new PopWin(my_name_="editor",win_name_="editor",win_title_="Editor for Tab: ",user_id=1, atm=atm_)
        //    editor.__init__();
        //    editor.set_win_frame_style("20", "650", "1000", "15%", "5%", "white")
        //    editor.set_acWinStatEventListeners(editor);
        //    }
        //    editor.set_acWinStat('block');
        //    atm_.editor=editor;

        //    alert(98765)
        //    alert(atm_.editor)
        //    alert(98765)
        }
  }.bind(atm_=this, event))

  this.add_delete_editor.appendChild(this.add_btn);
  this.add_delete_editor.appendChild(this.delete_btn);
  this.add_delete_editor.appendChild(this.editor_btn);
}

AdvancedTabsManager.prototype.setTabs = function()
{
  var dic_ = {"obj" : "AdvancedTabs", "atm": this.my_name, "app": this.my_app, "fun": "get_tabs_from_table", "params": {"name": ""}}
   //alert(JSON.stringify(dic_))
   //alert(this.activate_function_link_)
  $.post(this.activate_function_link_,{dic : JSON.stringify(dic_)},
      function(dic){
          //alert(JSON.stringify(dic))
          var result = dic["result"];
          if(result["manager"]==null){} else {atm_.content=result["manager"]}
          if(result["app_content"]==null){} else {atm_.app_content=result["app_content"]}
          //alert('JSON.stringify(atm_.content)')
          //alert(JSON.stringify(atm_.content))
          //alert('JSON.stringify(tabs)')
          //alert(JSON.stringify(tabs))
          var tabs = result["tabs"];
          for (var jj in tabs)
          {
            var k=tabs[jj]; var id_=k[0];
            atm_.tabs[id_]=new Tab(atm_, data=k[1], id=id_);
            atm_.tabs[id_].create_tab_pop_wins();
            atm_.set_active_tab(atm_.tabs[id_].btn)
          }
          try{atm_.tabs[id_].btn.click();} catch(er){}
     }.bind(atm_ = this)
 );
}

// This is a factory function. It recieve a parameters and produce an obj to return to the caller
AdvancedTabsManager.prototype.get_app_obj=function(obj_name, obj=null, is_new=false)
{
  //alert(obj_name + "  " + JSON.stringify(obj))
  var obj_dic="";
  for (var h in this.app_content){if(obj_name==this.app_content[h]["properties"]["obj_name"]){obj_dic=this.app_content[h];break;}}
  if(obj_dic==""){alert("There is no object "+obj_name+"."); return;}
  //alert(h + "  " + JSON.stringify(obj_dic))
  if(!(h in this.app_objs) || is_new==true){
    var n_=this.temp_number;this.temp_number+=1;

    var z=obj_dic["functions"]["constructor"]
    var z1=z.substring(0, z.indexOf('(')+1)
    var z2=z.substring(z.indexOf('(')+1)
    var z21=z2.substring(0, z2.indexOf('{')+1)
    var z22=z2.substring(z2.indexOf('{')+1)
    var temp_constractor=z1+"manager, "+z21+"this.my_name='"+obj_dic["properties"]["obj_name"]+"';this.atm=manager;"+z22
    var s=obj_dic["properties"]["obj_name"]+n_+'='+temp_constractor;eval(s);
    for(f in obj_dic["functions"])
    {if(f!="constructor"){s=obj_dic["properties"]["obj_name"]+n_+'.prototype.'+f+' = '+obj_dic["functions"][f];
       try{
       //alert(s)
       eval(s)}catch(er){alert("Error 9053: "+er)}}
    }
    var k="new "+obj_dic["properties"]["obj_name"]+n_+"(manager=this,obj)";
    //alert(k)
    var obj_=eval(k);
    obj_.atm=this;var s="this."+obj_name.toLowerCase()+"=obj_";eval(s)
    if(is_new==true){return obj_}
    this.app_objs[h]=obj_;
  } else {return this.app_objs[h]}
  return obj_
}

AdvancedTabsManager.prototype.delete_tab = function(tab_id_, tab_name_)
{this.tabs[tab_id_].btn.outerHTML="";this.tabs[tab_id_].content.outerHTML="";}

var clone_PopWinObjects={};
set_clone_PopWinObjects = function(PopWinObjects)
{
  clone_PopWinObjects = Object.assign({}, PopWinObjects)
}

AdvancedTabsManager.prototype.set_active_tab = function(btn)
{
  var tabcontent=document.getElementsByClassName("tabcontent");
  for(i=0;i<tabcontent.length;i++){if(tabcontent[i].tab.tab_name!="main_panel"){tabcontent[i].style.display='none';}}
  var tablinks=document.getElementsByClassName("tablinks");
  try{eval(btn.parent.tab_name+"__myclick__(called_tab=btn.parent, calling_tab=btn.parent)");} catch(er){alert("er9001: "+er)}
  for (i=0;i<tablinks.length;i++){
    try{
      tablinks[i].className=tablinks[i].className.replace(" active","");
      if(tablinks[i].parent.tab_name!=btn.parent.tab_name)
      {eval(tablinks[i].parent.tab_name+"__otherclick__(called_tab=tablinks[i].parent, calling_tab=btn.parent)");}
    } catch(er){alert('er9038: '+ er)}
  }
  try{for(w in this.active_tab.PopWinObjects){if(w=="editor"){ continue;};this.active_tab.PopWinObjects[w].temp_close_win()}} catch(er){}

  var click_event=new Event("click", {bubbles: true});
  try{
      this.active_tab=btn.parent;this.active_tab.content.style.display="block";btn.className+=" active";
      try{for(w in this.active_tab.PopWinObjects){if(w=="editor"){ continue;};

        if(this.active_tab.tab_active_link==null){this.active_tab.PopWinObjects[w].resume_win()} else{
          if(this.active_tab.PopWinObjects[w].popwin_content.link_dic["properties"]["container_id"]==this.active_tab.tab_active_link.content.link_number)
          {this.active_tab.PopWinObjects[w].resume_win()}
        }

      }} catch(er){}

      this.editor.main_menus["Tab"].btn.dispatchEvent(click_event);
      this.editor.set_title(this.editor.win_title_+this.active_tab.tab_name);
    } catch(er){}

    try{
       this.editor_btn.dispatchEvent(click_event);
       var tab_fpe_=this.editor.get_functions_properties_editor(
              this.active_tab,
              functions_dic=this.active_tab.tab_functions,
              functions_list_dic=this.tab["functions"],
              dic_properties=this.active_tab.tab_properties,
              settings_list=this.tab["settings_list"],
              attributes_list=this.tab["attributes_list"],
              tab_btn_name="Tab",properties_func=null,
              node_to_delete=null)
    } catch(er) {}
}

AdvancedTabsManager.prototype.get_tab = function(tab_name){for(id in this.tabs){if (this.tabs[id].tab_name==tab_name){return this.tabs[id]}}}

AdvancedTabsManager.prototype.get_pop_win = function(link_number){return this.pop_wins[link_number]}

// consider the following function
//var get_pop_win = function(link_number){return atm.get_pop_win(link_number)}


AdvancedTabsManager.prototype.truncate_model = function(model_name, app=null){
       var app_ = this.my_app; if(app!=null){app_=app}
       var dic_ = {"obj" : "AdvancedTabs", "atm": atm_.my_name, "app": this.my_app,
                   "fun": "truncate_model", "params": {"model_name": model_name, "app":app_}}
       // alert(JSON.stringify(dic_))
              $.post(atm_.activate_function_link_,
              {
                dic : JSON.stringify(dic_)
              },
              function(dic){
                //alert(JSON.stringify(dic))
             }.bind(atm=atm_))
}

// this is a Factory function which return object based on the dic.
AdvancedTabsManager.prototype.get_obj = function(tab, dic)
{
// if(dic["obj_name"]=="acGrades"){
   //alert("90111 "+tab.tab_name)
//   alert(JSON.stringify(dic))
// }
 var s='function '+dic["obj_name"]+dic["properties"]["obj_number"]+'(atm_, tab_, dic_)'
 s+='{'
 //s+='this.my_type="'+dic["obj_name"]+'";this.my_element="'+dic["element_name"]+'";';
 s+=dic["obj_type"]+'.call(this);'
 s+='this.tab=tab_;this.atm=atm_;this.editor=this.atm.editor;this.data=dic_;var ps=this.atm.get_obj_functions_settings_attributes(dic_);'
 s+='this.settings=ps["settings"];'
 s+='this.functions=ps["functions"] ;'
 s+='this.attributes=ps["attributes"] ;'
 s+='try{this.creator=new '+dic["obj_name"]+'Creator(this)} catch(er){};'
 s+='};'
//     if(dic["obj_name"]=="acRichText"){
//       alert(s)
//     }
 try{eval(s)} catch(er){alert("er9002: "+er)}
 s=dic["obj_name"]+dic["properties"]["obj_number"]+'.prototype = Object.create('+dic["obj_type"]+'.prototype);'
 //alert(s);
 try{eval(s)} catch(er){alert("er903: "+er)}
    //--
    // if(dic["obj_name"]=="acPivot")
    // {
    // //alert(JSON.stringify(dic))
    // s=dic["obj_name"]+dic["properties"]["obj_number"]+'.prototype.create_editor = function()'
    // s+='{'
    // s+=' var container_id=this.data["container_id"];';
    // s+=' var obj_number=this.data["properties"]["obj_number"];';
    // s+=' try{var properties_func_=null;properties_func_=this.creator.editor_properties_func}catch(er){};';
    // s+=' try{'
    // // s+='alert(JSON.stringify(this.data["properties"]));'
    // s+='  var fpe_=this.editor.get_functions_properties_editor(this.tab, this.data["functions"], this.functions,'
    // s+='   this.data["properties"], this.settings,this.attributes, tab_btn_name="Plugin", properties_func=properties_func_,'
    // s+='   node_to_delete=".tab_objects["+container_id+"]["+obj_number+"]"';
    // s+='  )';
    // s+=' } catch(er) {alert("er 2070: "+er)};'
    // s+='}'
    // //alert(s);
    // try{eval(s)}catch(er){alert(er)}
    // } else {
 s=dic["obj_name"]+dic["properties"]["obj_number"]+'.prototype.create_editor = function()'
 s+='{'
 s+=' var container_id=this.data["container_id"];';
 s+=' var obj_number=this.data["properties"]["obj_number"];';
 s+=' try{var properties_func_=null;properties_func_=this.creator.editor_properties_func}catch(er){};';
 s+=' try{'
 //s+='alert(JSON.stringify(this.data["properties"]));'
 s+='  var fpe_=this.editor.get_functions_properties_editor(this.tab, this.data["functions"], this.functions,'
 s+='   this.data["properties"], this.settings,this.attributes, tab_btn_name="Plugin", properties_func=properties_func_,'
 s+='   node_to_delete=".tab_objects["+container_id+"]["+obj_number+"]"';
 s+='  )';
 s+=' } catch(er) {alert("er 2071: "+er)};'
 s+='}'

// if(dic["obj_name"]=="acGrades"){
// alert(s);
// }

 eval(s);
// }
 //--
 s = 'new '+dic["obj_name"]+dic["properties"]["obj_number"]+'(atm_=this, tab_=tab, dic_=dic)'
 //alert(s)
 try{obj_=eval(s)
 }catch(er){alert(er)}
 return obj_
}

AdvancedTabsManager.prototype.get_rich_text_editor=function(div_to_edit){
       //var d_container=document.createElement("div");
       //d_container.setAttribute("class", "container")

       var d_options=document.createElement("div");d_options.setAttribute("class", "options")
       var ss="position:absolute;border-radius:10px;border-style:solid;border-width:1px;border-color:blue;background-color:white;z-index:999999;"
       ss+="padding: 5px;left:"+div_to_edit.getAttribute("rich_text_x")+"px;top:"+div_to_edit.getAttribute("rich_text_y")+"px;width:440px";
       d_options.setAttribute("style", ss)

       //d_container.appendChild(d_options);
       //d_container.appendChild(div_to_edit);

       var f_=function(id, s_class){
         var b=document.createElement("button");b.setAttribute("id", id);b.setAttribute("class", s_class+" ebutton");
         var i=document.createElement("i");i.setAttribute("class", t_class);
         b.appendChild(i);return b;
       }
        //main logic
        const modifyText = (command, defaultUi, value) => {
          //execCommand executes command on selected text
          document.execCommand(command, defaultUi, value);
        };
        //Highlight clicked button
        const highlighter = (className, needsRemoval) => {
          className.forEach((button) => {
            button.addEventListener("click", () => {
              //needsRemoval = true means only one button should be highlight and other would be normal
              if (needsRemoval) {
                let alreadyActive = false;
                //If currently clicked button is already active
                if (button.classList.contains("active")) {
                  alreadyActive = true;
                }
                //Remove highlight from other buttons
                highlighterRemover(className);
                if (!alreadyActive) {
                  //highlight clicked button
                  button.classList.add("active");
                }
              } else {
                //if other buttons can be highlighted
                button.classList.toggle("active");
              }
            });
          });
        };

        var b=f_(id="bold", s_class="option-button format", t_class="fa-solid fa-bold");d_options.appendChild(b);
        b=f_(id="italic", s_class="option-button format", t_class="fa-solid fa-italic");d_options.appendChild(b);
        b=f_(id="underline", s_class="option-button format", t_class="fa-solid fa-underline");d_options.appendChild(b);
        b=f_(id="strikethrough", s_class="option-button format", t_class="fa-solid fa-strikethrough");d_options.appendChild(b);
        b=f_(id="superscript", s_class="option-button script", t_class="fa-solid fa-superscript");d_options.appendChild(b);
        b=f_(id="subscript", s_class="option-button script", t_class="fa-solid fa-subscript");d_options.appendChild(b);
        b=f_(id="insertOrderedList", s_class="option-button", t_class="fa-solid fa-list-ol");d_options.appendChild(b);
        b=f_(id="insertUnorderedList", s_class="option-button", t_class="fa-solid fa-list");d_options.appendChild(b);
        b=f_(id="undo", s_class="option-button", t_class="fa-solid fa-rotate-left");d_options.appendChild(b);
        b=f_(id="redo", s_class="option-button", t_class="fa-solid fa-rotate-right");d_options.appendChild(b);
        b=f_(id="createLink", s_class="adv-option-button", t_class="fa fa-link");d_options.appendChild(b);
        b.addEventListener("click", () => {
          let userLink = prompt("Enter a URL");
          //if link has http then forward directly else add https
          if (/http/i.test(userLink)) {
            modifyText(linkButton.id, false, userLink);
          } else {
            userLink = "http://" + userLink;
            modifyText(linkButton.id, false, userLink);
          }
        });

        b=f_(id="unlink", s_class="option-button", t_class="fa fa-unlink");d_options.appendChild(b);

        b=f_(id="justifyLeft", s_class="option-button align", t_class="fa-solid fa-align-left");d_options.appendChild(b);
        b=f_(id="justifyCenter", s_class="option-button align", t_class="fa-solid fa-align-center");d_options.appendChild(b);
        b=f_(id="justifyRight", s_class="option-button align", t_class="fa-solid fa-align-right");d_options.appendChild(b);
        b=f_(id="justifyFull", s_class="option-button align", t_class="fa-solid fa-align-justify");d_options.appendChild(b);

        b=f_(id="indent", s_class="option-button spacing", t_class="fa-solid fa-indent");d_options.appendChild(b);
        b=f_(id="outdent", s_class="option-button spacing", t_class="fa-solid fa-outdent");d_options.appendChild(b);
        var s=document.createElement("select");s.setAttribute("id","formatBlock");s.setAttribute("class","adv-option-button");d_options.appendChild(s);
        for(var j=1;j<7;j++){var o=document.createElement("option");
        o.setAttribute("value", "H"+j);o.innerHTML="H"+j;s.appendChild(o);}
        //--
        var s=document.createElement("select");s.setAttribute("id","fontName");s.setAttribute("class","adv-option-button");d_options.appendChild(s);
        let fontList = ["Arial","Verdana","Times New Roman","Garamond","Georgia","Courier New","cursive"];
        fontList.map((value) => {let o=document.createElement("option");o.value = value;o.innerHTML = value;s.appendChild(o);});

        var s=document.createElement("select");s.setAttribute("id","fontSize");s.setAttribute("class","adv-option-button");d_options.appendChild(s);
        for (let i=1;i<=7;i++){let o=document.createElement("option");o.value=i;o.innerHTML=i;s.appendChild(o);};s.value = 3;
        //--
        var d=document.createElement("div");d.setAttribute("class","input-wrapper");d_options.appendChild(d);
        var i=document.createElement("input");i.setAttribute("id","foreColor");i.setAttribute("type","color");
        i.setAttribute("class","adv-option-button");d.appendChild(i);
        var l=document.createElement("label");l.setAttribute("for","foreColor");l.innerHTML="fC";d.appendChild(l);

        var d=document.createElement("div");d.setAttribute("class","input-wrapper");d_options.appendChild(d);
        var i=document.createElement("input");i.setAttribute("id","backColor");i.setAttribute("type","color");
        i.setAttribute("class","adv-option-button");d.appendChild(i);
        var l=document.createElement("label");l.setAttribute("for","backColor");l.innerHTML="bgC";d.appendChild(l);

        var b=f_(id="save_save_", s_class="option-button format", t_class="fas fa-window-close fa-lg");d_options.appendChild(b);
        b.d_options=d_options;b.div_to_edit=div_to_edit;
        b.addEventListener("click", (event) => {
         var e=event.target;if(e.tagName=="I"){e=e.parentNode;}
         e.div_to_edit.setAttribute("contenteditable", "");e.d_options.outerHTML="";
        });

        let optionsButtons = d_options.querySelectorAll(".option-button");
        //For basic operations which don't need value parameter
        optionsButtons.forEach((button) => {
          button.addEventListener("click", () => {
            modifyText(button.id, false, null);
          });
        });

        let advancedOptionButton = d_options.querySelectorAll(".adv-option-button");
        //options that require value parameter (e.g colors, fonts)
        advancedOptionButton.forEach((button) => {
          button.addEventListener("change", () => {
            modifyText(button.id, false, button.value);
          });
        });

        const highlighterRemover = (className) => {
          className.forEach((button) => {
            button.classList.remove("active");
          });
        };

        let alignButtons = d_options.querySelectorAll(".align");
        let spacingButtons = d_options.querySelectorAll(".spacing");
        let formatButtons = d_options.querySelectorAll(".format");
        let scriptButtons = d_options.querySelectorAll(".script");
        highlighter(alignButtons, true);
        highlighter(spacingButtons, true);
        highlighter(formatButtons, false);
        highlighter(scriptButtons, true);

        return d_options;
    }

AdvancedTabsManager.prototype.get_obj_functions_settings_attributes = function(dic_)
{
 //alert(JSON.stringify(dic_));
 var ps={"settings": {"width":[], "x":[], "y":[], "title":[], "obj_number":[]},
         "attributes":{},
         "functions":["onclick"]};
 var dic__=this.buttons[dic_["parent_obj_name"]]["sub_buttons"][dic_["element_name"]]

 //alert(JSON.stringify(dic_));
 //alert(JSON.stringify(dic__));

 //for (a in dic_["attributes"]){if (!ps["attributes"].includes(a)){ps["attributes"].push(a)}};
 //for (s in dic_["setting"]){if (!ps["settings"].includes(s)){ps["settings"].push(s)}};
 //--
 for (k in dic__["attributes"]){if(!(k in ps["attributes"])){ps["attributes"][k]=dic__["attributes"][k]}}
 for (k in dic__["setting"]){
 if(!(k in ps["settings"])){ps["settings"][k]=dic__["setting"][k]}}
 //for (i in dic__["setting"]){if (!ps["settings"].includes(dic__["setting"][i])){ps["settings"].push(dic__["setting"][i])}};
 for (i in dic__["functions"]){if (!ps["functions"].includes(dic__["functions"][i])){ps["functions"].push(dic__["functions"][i])}};
 for (f in dic_["functions"]){if (!ps["functions"].includes(f)){ps["functions"].push(f)}};
// alert(JSON.stringify(ps));
 //if (!ps["attributes"].includes(dic__["attributes"][i]))
 //{ps["attributes"].push(dic__["attributes"][i])}};
 //--
 //alert(JSON.stringify(ps));
 return ps;
}

AdvancedTabsManager.prototype.save = function()
{
  try{
    var tab_content = {
                      "tab_content_link_dic": this.active_tab.tab_content_link_dic,
                      "functions": this.active_tab.tab_functions,
                      "nav_links": this.active_tab.tab_nav_links,
                      "properties": this.active_tab.tab_properties,
                      "objects": this.active_tab.tab_objects,
                      "tab_name": this.active_tab.tab_name,
                      "tab_order": this.active_tab.tab_order,
                      "tab_title": this.active_tab.tab_title,
                      "tab_type": this.active_tab.tab_type,
                      "tab_pop_win_buttons": this.active_tab.tab_pop_win_buttons
                    }
  } catch(er){alert("er9004: "+er)}
  var dic_={"obj":"AdvancedTabs","atm":this.my_name, "app": this.my_app, "fun":"save_content",
            "params": {"atm_content": this.content,
                       "app_content": this.app_content,
                       "tab_content": tab_content,
                       "tab_name": this.active_tab.tab_name,
                       "tab_order": this.active_tab.tab_properties["tab_order"],
                       "tab_id": this.active_tab.tab_id}}
      //alert("atm.save:active_tab.tab_pop_win_buttons\n-------\n"+JSON.stringify(this.active_tab.tab_pop_win_buttons))
      //alert(JSON.stringify(dic_))
      //alert(this.activate_function_link_)
     $.post(this.activate_function_link_,
          {dic : JSON.stringify(dic_)},
          function(dic){
            if(dic["status"]!="ok"){alert(JSON.stringify(dic["result"]))
            }
            else{  //alert("ok")
            }
          });
}

AdvancedTabsManager.prototype.logmein = function(dic_)
{
   dic_["app"]=this.my_app
   //alert(this.logmein_link_)
   //alert(JSON.stringify(dic_))
   $.post(this.logmein_link_,
     {dic : JSON.stringify(dic_)},
     function(dic){
       if(dic["status"]=="ok"){
         try{location.reload();} catch(er){alert(er)}
       }
     });
}

AdvancedTabsManager.prototype.activate_obj_function = function(call_back_fun, dic_, html_obj)
{
  dic_["company_obj_id"]=this.company_obj_id;
     //alert("activate_obj_function\n"+JSON.stringify(dic_))
     //alert(this.activate_obj_function_link_)
     //alert(call_back_fun)

     $.post(atm.activate_obj_function_link_,
          {dic : JSON.stringify(dic_)},
          function(dic){
          //alert(JSON.stringify(dic))
            if(dic["status"]=="ok"){
                //alert("ok\n"+JSON.stringify(dic))
                try{
                  call_back_fun(dic["result"], html_obj);
                  html_obj.data=dic["result"];
                  } catch(er){}
            } else {alert("Error2020 getting data for select.")}
          });
}

AdvancedTabsManager.prototype.save_data = function(html_obj, dic_, is_json_data=false)
{ //alert("AdvancedTabsManager.prototype.save_data")
  dic_["app"]=this.my_app;
  dic_["company_obj_id"]=this.company_obj_id;
  //alert("90112:\n"+JSON.stringify(dic_));
  //alert(html_obj.outerHTML)
  //alert("90122 "+this.update_field_link_)
  $.post(this.update_field_link_,
          {dic : JSON.stringify(dic_)},
          function(dic){
            if(dic["status"]!="ok")
            {//alert("Data was not saved.")
            }
            else
            {
              var record_level="record_id"; if(is_json_data==true){record_level="parent_id"}
              try{var element_id_=null;var element_id_=dic["element_id"];} catch(er){alert(er)}
              try{if(element_id_!=null){try{html_obj_=getEBI(html_obj_.element_id_);} catch(er){}}} catch(er){}
              html_obj_.setAttribute(record_level, dic["record_id"])
              var creator_number=html_obj_.getAttribute("my_creator_number")
              try{if(!(creator_number==null)){get_creator(creator_number).on_record_created_deleted(html_obj_);}} catch(er){alert(er)}
              try{
                 var f=dic_["for_refresh_my_tables"]["f"]; var v=dic_["for_refresh_my_tables"]["v"];
                 var container=getEBI("content_"+dic_["for_refresh_my_tables"]["container_id"])
                 container.my_creator_obj.refresh_my_tables(f, v);
              }catch(er){}
            }
          }.bind(html_obj_=html_obj));
}

AdvancedTabsManager.prototype.get_data = function(call_back_fun, dic_, tbody_)
{
//  alert("1001213  "+JSON.stringify(dic_));
  dic_["app"]=this.my_app;
  if(dic_["company_obj_id"]==null){dic_["company_obj_id"]=this.company_obj_id;}
  //for(var k in dic_){alert(dic_[k])}
  if(dic_["from_fun"]!=null){if(dic_["from_fun"]=="acReport.prototype.set_data"){alert(dic_["from_fun"]+"\n\n"+JSON.stringify(dic_))}}
  //alert("9090-9090 AdvancedTabsManager.prototype.get_data: "+this.get_data_link_)
  call_back_fun.atm=this
  $.post(this.get_data_link_,
          {dic : JSON.stringify(dic_)},
          function(data){
          try{
          if(dic_["from_fun"]!=null){if(dic_["from_fun"]=="acReport.prototype.set_data")
          {alert(dic_["from_fun"]+"\n\n"+JSON.stringify(data))}}

           if(data["status"]!="ok"){return}
           //alert("9001-9001-100 data in atm "+JSON.stringify(data));
           //try{alert(call_back_fun)} catch(er){"call_back 9001-9001-200\n"+alert(er)}
           try{call_back_fun(data["dic"], tbody_)} catch(er){"call_back 9001-9001-300\n"+alert(er)}
           //alert("after call_back_fun")
          } catch(er){alert("ERROR\n9090-9090-1 AdvancedTabsManager.prototype.get_data:\n"+JSON.stringify(dic_)+"\n"+er)}
          try{tbody_.data=data["dic"];} catch(er){alert("ERROR\n9090-9090-2 AdvancedTabsManager.prototype.get_data:\n"+JSON.stringify(dic_)+"\n"+er)}
           //alert("9005"+JSON.stringify(data));

          });
}

AdvancedTabsManager.prototype.get_data_json = function(call_back_fun, dic_, return_obj)
{
  //alert("100123  "+JSON.stringify(dic_));
  dic_["app"]=this.my_app;
  if(dic_["company_obj_id"]==null){dic_["company_obj_id"]=this.company_obj_id;}
  //alert("1001231111 "+JSON.stringify(dic_));
  //alert(call_back_fun)
  //alert(this.get_data_json_link_)
  call_back_fun.atm=this;
  $.post(this.get_data_json_link_,
          {dic : JSON.stringify(dic_)},
          function(data){
          try{
           //alert("9001"+JSON.stringify(data));
           if(data["status"]!="ok"){return}
           call_back_fun(data["dic"], return_obj)
           return_obj.data=data["dic"];
           //alert("9005"+JSON.stringify(data));
           } catch(er){alert(er)}
          });
}

AdvancedTabsManager.prototype.get_list = function(call_back_fun, dic_, html_obj)
{
  dic_["company_obj_id"]=this.company_obj_id;
  try{dic_["params"]["company_obj_id"]=this.company_obj_id;} catch (er) {}
//  alert("AdvancedTabsManager.prototype.get_list dic=\n"+this.activate_function_link_+"\n"+JSON.stringify(dic_))
     $.post(this.activate_function_link_,
          {dic : JSON.stringify(dic_)},
          function(dic){
            if(dic["status"]=="ok"){
                //alert(JSON.stringify(dic["result"]))
                call_back_fun(dic["result"], html_obj);
                html_obj.data=dic["result"];try{eval(html_obj.run_eval);
                //alert(html_obj.run_eval)
                try{
                     //alert(html_obj.is_call_on_change==true)
  //alert("9\n="+html_obj.is_call_on_change+"=\n"+html_obj.outerHTML)
                     if(html_obj.is_call_on_change=="true"){
  //alert("10\n="+html_obj.is_call_on_change+"=\n"+html_obj.outerHTML)
                      try{var ce = new Event("change", {bubbles: true});html_obj.dispatchEvent(ce);} catch (er){alert("er9029: "+ er)}
                     }
                     html_obj.is_call_on_change=true;
                    } catch(er){}
                } catch(er){}

            } else {alert("Error getting data for select.")}
          });
}

AdvancedTabsManager.prototype.get_next_obj_number=function(){
this.content["last_obj_number"]+=1;
return this.content["last_obj_number"]
}

AdvancedTabsManager.prototype.set_obj_to_copy=function(dic){
 this.new_obj_to_create = dic;
}

AdvancedTabsManager.prototype.delete_record_from_table=function(call_back_fun, dic_, html_obj=null){
  //alert("1001213  "+JSON.stringify(dic_));
  dic_["app"]=this.my_app;
  if(dic_["company_obj_id"]==null){dic_["company_obj_id"]=this.company_obj_id;}
  $.post(this.delete_record_from_table_link_,
          {dic : JSON.stringify(dic_)},
          function(data){
          try{
           //alert("9001-78-200"+JSON.stringify(data));
           if(data["status"]!="ok"){return}
           call_back_fun(data, html_obj)
           } catch(er){alert("9090-120 AdvancedTabsManager.delete_record_from_table:\n"+er)}
          });
}

AdvancedTabsManager.prototype.general_data = function(call_back_fun, dic_, obj_)
{
  //alert("In the function: AdvancedTabsManager.prototype.general_data")
  dic_["app"]=this.my_app;
  dic_["company_obj_id"]=this.company_obj_id;
  call_back_fun.atm=this
  //alert("90112:\n"+JSON.stringify(dic_));
  //alert(this.general_data_)
  $.post(this.general_data_, {dic : JSON.stringify(dic_)},
          function(data){
            if(data["status"]!="ok") {alert("Error in General_Data.")} else {call_back_fun(data["json"], obj_)}
          })
}

// -- FunctionsPropertiesEditor --
function FunctionsPropertiesEditor(tab, functions_dic, functions_list_dic, properties_dic, settings_list,
                                   attributes_list, tab_btn_name="PopWin", properties_func=null, node_to_delete=null)
{
  //alert(JSON.stringify(functions_dic))
  //alert(JSON.stringify(functions_list_dic))
  //alert("555555555: "+tab.tab_name+" - "+JSON.stringify(properties_dic))
  //alert(JSON.stringify(settings_list))
  //alert(JSON.stringify(attributes_list))
  //--
  tab.parent.editor.main_menus[tab_btn_name].btn.click();
  tab.parent.editor.win_content.innerHTML="";
  var lef_nav=document.createElement("div");
  lef_nav.setAttribute("class", "tab");
  var new_del_table=document.createElement("table");
  //new_del_table.setAttribute("style","width:40%")
  var tr=document.createElement("tr");new_del_table.appendChild(tr);
  var tdd=document.createElement("td");var del_btn=document.createElement("button");del_btn.innerHTML="Del Func";tdd.appendChild(del_btn);tr.appendChild(tdd);
  var tdn=document.createElement("td");var new_btn=document.createElement("button");new_btn.innerHTML="New Func";tdn.appendChild(new_btn);tr.appendChild(tdn);
  lef_nav.appendChild(new_del_table);
  //--
  new_btn.onclick= function (){
    var fun_name_ = prompt("Enter name for new function:" , '');
    if(fun_name_ == '') {alert("Please enter a function name"); return;}
    if(tbn=="Tab")
    {
     var tab_name=tab_.tab_name;
     functions[tab_name+"_"+fun_name_]=tab_name+"_"+fun_name_+"=function(obj){\ntry{\n\n} catch(er){alert('er9026: '+ er)}}";
     tab_.active_function = tab_.tab_name+"_"+fun_name_;
     try{
        tab_.parent.save();
        var click_event = new Event("click", {bubbles: true});tab_.btn.dispatchEvent(click_event);
      } catch (er){alert("er9029: "+ er)}
    } else {
     functions[fun_name_]="function (obj){\ntry{\n\n} catch(er){alert('er9005: '+er)}}";
     editor.tab_obj_.parent.save();
    }
  }.bind(tab_=tab, tbn=tab_btn_name, editor=tab.parent.editor, functions=functions_dic, event)
  del_btn.onclick= function (){
    event.preventDefault();
    var f=editor.tab_obj_.active_function;
    var fun_name_ = prompt("Are you sure you want to delete function "+f+". If so, type Yes:" , 'No');
    if(fun_name_ != 'Yes') {return;};
    delete functions[f]
    tab_.parent.save();
    if(tbn=="Tab")
    {try{var click_event = new Event("click", {bubbles: true});tab_.btn.dispatchEvent(click_event);} catch (er){alert("er9029: "+ er)}
    } else {alert("function "+f+" was deleted.")}
  }.bind(tab_=tab, tbn=tab_btn_name, editor=tab.parent.editor, functions=functions_dic, event)
  //--
  var nav_div=document.createElement("div");
  var tab_content = document.createElement("textarea");
  nav_div.onclick=function(){
    //alert(JSON.stringify(editor_.tab_obj_.tab_functions))
    var e=event.target;
    try{var cc=e.getAttribute("class");if(cc!="funtablinks"){return;}} catch(er){}
    var f=event.target.innerHTML;
    editor.tab_obj_.active_function=f;
    var fun=null;
    try{fun=functions_[f]} catch(er){}
    if(fun==null){fun="function (event){\ntry{\n\n} catch(er){alert('err 9016-1: '+er)}\n}";functions_[f]=fun;}
      tab_content_.innerHTML=functions_[f];
      tab_content_.setAttribute("fun_name",f);
      var funtablinks = document.getElementsByClassName("funtablinks");
      for (var i=0;i<funtablinks.length;i++){funtablinks[i].className=funtablinks[i].className.replace(" active","");};
      event.target.className += " active";
       //alert(editor_.tab_content.outerHTML)
    }.bind(editor=tab.parent.editor, tab_content_=tab_content, functions_=functions_dic, event);
  nav_div.btns={};

  var fs__=functions_list_dic;var tfs_=Object.keys(functions_dic);var fs_=[];

  for(i in fs__){if(!(fs_.includes(fs__[i]))){fs_.push(fs__[i])}}
  for(i in tfs_){if(!(fs_.includes(tfs_[i]))){fs_.push(tfs_[i])}}
  for(i in fs_)
  {var f=fs_[i];nav_div.btns[f]=document.createElement("button");
      nav_div.btns[f].setAttribute("class", "funtablinks");
      nav_div.btns[f].innerHTML=f;nav_div.appendChild(nav_div.btns[f]);
  }

   lef_nav.appendChild(nav_div);
   tab.parent.editor.win_content.appendChild(lef_nav);
   //--
   tab_content.setAttribute("class", "tab_textarea");
   tab_content.onchange= function (){
    //alert(event.target.value)
    //alert(event.target.getAttribute("fun_name"))
    //alert(JSON.stringify(functions__))

     functions__[event.target.getAttribute("fun_name")]=event.target.value;
     editor.tab_obj_.parent.save();
    }.bind(editor=tab.parent.editor, functions__=functions_dic, event)
   tab.parent.editor.win_content.appendChild(tab_content);
   //--
   var tab_properties_ = document.createElement("div");
   if(node_to_delete!=null){
      var btn=document.createElement("button");btn.innerHTML="Delete Obj";btn.tab=tab;
      btn.onclick=function(event){
        var link_ = prompt("Are you sure you want to remove the obj? if so type Yes:",'No');
        if(link_=='Yes')
        {
         //alert(tab_.tab_name);
         try{var s='delete tab_'+node_to_delete_;eval(s)} catch(er) {}
         try{tab_.parent.save();} catch(er) {alert("error 2099: "+ er)}
        }
      }.bind(tab_=tab, node_to_delete_=node_to_delete)
      tab_properties_.appendChild(btn);
    }

   var cbtn=document.createElement("button");cbtn.innerHTML="Copy Obj";cbtn.tab=tab;
   cbtn.onclick=function(event){
         var clone = JSON.parse(JSON.stringify(this.tab.active_obj.data))
         //alert("original\n\n"+JSON.stringify(this.tab.active_obj.data))
         clone["container_id"]=""
         clone["properties"]["obj_number"]=""
         this.tab.parent.set_obj_to_copy(clone);
      }
   tab_properties_.appendChild(cbtn);
   var table = document.createElement("table");var tr=document.createElement("tr");table.appendChild(tr);
   var thp=document.createElement("th");thp.innerHTML="Property";thp.setAttribute("style","width:10%;text-align:center;");
   tr.appendChild(thp);
   var thv=document.createElement("th");thv.innerHTML="Value";thv.setAttribute("style","width:10%;text-align:center;");
   tr.appendChild(thv);
   tab_properties_.appendChild(table);

   var pp_={}
   for (k in settings_list){if (!(k in pp_)){pp_[k]=settings_list[k]}}
   for (k in attributes_list){if (!(k in pp_)){pp_[k]=attributes_list[k]}}

   //alert(JSON.stringify(properties_dic));
   //for (k in properties_dic){if (!(k in pp_)){pp_[k]=properties_dic[k]}}
   //alert(JSON.stringify(settings_list));
   //alert(JSON.stringify(attributes_list));
   //alert(JSON.stringify(pp_));

   for (k in pp_)
   {
     var tr=document.createElement("tr");table.appendChild(tr);
     var td=document.createElement("td");td.innerHTML=k;tr.appendChild(td);
     var td=document.createElement("td");
     var l=pp_[k];
     if(l.length == 0){
       var input=document.createElement("input");input.setAttribute("size","10");
       if(k.includes("color")){input.setAttribute("type","color")};
     } else {
       var input=document.createElement("select");input.setAttribute("style", "width:100px;");
       //if(k.includes('color')){input.setAttribute("type","color")}
       for (j in l){var o = document.createElement("option");o.value=l[j];o.innerHTML=l[j];input.appendChild(o);}
     }
     input.setAttribute("property",k);td.appendChild(input);
     //input.setAttribute("size","10");
     //input.setAttribute("property",s);
     try{if(properties_dic[k]==null){} else{input.value=properties_dic[k]}} catch(er){alert("er9008: "+er)}
     tr.appendChild(td);
   }

   tab_properties_.addEventListener("change", function(){
     var p=event.target;var property=p.getAttribute("property");var v=p.value;
     //alert(property); alert(v); alert(JSON.stringify(properties));
     properties[property]=v;
     //alert(JSON.stringify(properties));
      editor.tab_obj_.parent.save();
   }.bind(editor=tab.parent.editor, properties=properties_dic, event))
   tab_properties_.setAttribute("class", "com_setting");
   tab.parent.editor.win_content.appendChild(tab_properties_);
   //alert(properties_func)
   try{if(properties_func!=null){properties_func(tab, tab_properties_)}} catch(er) {alert("error 206: "+er)}
}


//-- AccountingObj --
function AccountingObj(parent, cofa_dic=null)
{
 //alert(JSON.stringify(cofa_dic))
 this.parent=parent;
 this.cofa=null;
 this.accounting_data=null;
 this.data_for_report={};
 if(cofa_dic!=null){this.set_cofa(cofa_dic)}
}

AccountingObj.prototype.set_cofa = function(cofa_dic)
{
 var f = function(result, account_math, dic, c=[])
 {
  for(var k in dic)
  {
   if(!isNaN(parseInt(k))){for(j in c){if(!(c[j] in result)){result[c[j]]={}};result[c[j]][k]=dic[k];}}
   else{
      var ss_=k.split("[")
      if(ss_.length>1)
      {
       dic[ss_[0]] = clone_dic(dic[k]); delete dic[k];
       k=ss_[0];var s_=ss_[1].split("]");account_math[k]={"sign":s_[0],"account":s_[1]}
      }
      var cc = [k].concat(c);f(result, account_math,dic[k],cc);
    }
  }
 }
 this.cofa=cofa_dic;
 this.accounts={};this.accounts_math={}

 f(this.accounts, this.accounts_math, this.cofa)

 //alert("this.accounts_math=\n"+JSON.stringify(this.accounts_math))
 //alert("The following is the lists of accounts")
 //for(k in this.accounts){alert(k+"\n\n"+JSON.stringify(this.accounts[k]))}
}

AccountingObj.prototype.set_time_dim = function(html_obj)
{
  var fun = function(data, html_obj)
  {
    html_obj.innerHTML = "Process finished with last date: "+data["last_date"]
  }
  var dic={"obj": "AccountingObj", "atm": atm.my_name, "app": atm.my_app,
           "fun": "set_time_dimension","params": {"app": atm.my_app, "table_name": "TimeDim"}}
  this.parent.get_list(fun,dic,html_obj)
}

AccountingObj.prototype.update_trial_balance = function(html_obj, start_date, end_date)
{
 var ss = start_date.split("-"); start_date=ss[0]+ss[1]+ss[2]
 var ss = end_date.split("-"); end_date=ss[0]+ss[1]+ss[2]
  //alert(start_date + " " + end_date)

  var fun = function(data, html_obj)
  {
    html_obj.innerHTML = "Process finished with start date: "+data["start date"] + " "+ data["end date"]
  }
  var dic={"obj": "AccountingObj", "atm": atm.my_name, "app": atm.my_app,
           "fun": "update_trial_balance",
           "params": {"app": atm.my_app,
                      "start_date": start_date,
                      "end_date":end_date,
                      "gld_table_name":"GeneralLedgerDetail",
                      "tb_table_name":"TrialBalance"
                     }}
   //alert(JSON.stringify(dic));
  this.parent.get_list(fun,dic,html_obj)
}

AccountingObj.prototype.set_accounting_data = function(data, time_dim, type="balance_sheet")
{
  this.accounting_data=data;
  var total={};for(var t_ in time_dim){total[t_]={"value":0, "color":"blue"}}
  var al_=this.accounts["Balance Sheet"]
  var ai_=this.accounts["Income"]
  var am=this.accounts_math
  var ff=function(cofa, data, spc="",al=null,am)
  {
   var htm="";var total_inner=clone_dic(total);var total_outer=clone_dic(total);var total_inner_outer=clone_dic(total)
   for(var k in cofa)
   {
   //alert("-- "+k+" --")
    if(!isNaN(parseInt(k)))
    {htm+="<tr><th style='text-align:left'>"+spc+k+":"+al[k]+"</th>"
     for(var t_ in time_dim)
     {try{total_inner[t_]["value"]=1*total_inner[t_]["value"]+1*data[k][t_]["value"]} catch(er){}
      htm+="<td style='text-align:right'>";try{htm+=nice_number(data[k][t_]["value"])}catch(er){};htm+="</td>"}
     htm+="</tr>"
    }
    else
    {
      htm+="<tr><th style='text-align:left'>"+spc+k+"</th>";for(var h in time_dim){htm+="<td></td>"};htm+="</tr>";
      //alert("===1 "+k+" ===")
      var t_htm=ff(cofa[k], data,spc+"&nbsp;&nbsp;",al=al,am);
      //alert("===2 "+k+" ===")

      htm+=t_htm[1];
      htm+="<tr><th style='text-align:left'>"+spc+"&nbsp;&nbsp;Total "+k+"</th>"
      for(var h in time_dim){
         var s_color="black";try{s_color=t_htm[0][h]["color"]}catch(er){}
         var value_="";try{value_=t_htm[0][h]["value"]} catch(er){}
         htm+="<td style='color:"+s_color+";text-align:right'>"+nice_number(value_)+"</td>"
      };htm+="</tr>";

      var sign=1; var add_account="";
      if(k in am){if(am[k]["sign"]=="-"){sign=-1};add_account=am[k]["account"]}

    //alert("50 inner k= "+k+"\n"+JSON.stringify(total_inner)+"\nt_htm[0]\n"+JSON.stringify(t_htm[0])+"\nouter\n"+JSON.stringify(total_outer))

      for(var h in time_dim){
         var value_=0;try{value_=t_htm[0][h]["value"];
          total_outer[h]["value"]+=sign*t_htm[0][h]["value"]
          } catch(er){}
      }

    //alert("5 inner k= "+k+"\n"+JSON.stringify(total_inner)+"\nouter\n"+JSON.stringify(total_outer))

      if(add_account!=""){
        htm+="<tr><th style='text-align:left'>"+add_account+"</th>";
        for(var h in time_dim){
           var s_color="black";try{s_color=total_outer[h]["color"]}catch(er){};
           var value_="";try{value_=total_outer[h]["value"]+total_inner[h]["value"]}catch(er){}
           htm+="<td style='color:"+s_color+";text-align:right'>"+nice_number(value_)+"</td>"
        };htm+="</tr>";
      }

    }

    //alert("inner k= "+k+"\n"+JSON.stringify(total_inner)+"\nouter\n"+JSON.stringify(total_outer))


   }

    //alert("End End inner k= "+k+"\n"+JSON.stringify(total_inner)+"\nouter\n"+JSON.stringify(total_outer))

    for(var h in time_dim){
         try{total_inner_outer[h]["value"]=total_inner[h]["value"]+total_outer[h]["value"]} catch(er){}
      }

    //alert("End End inner_outer\n"+JSON.stringify(total_inner_outer))

   return [total_inner_outer, htm];
  }

  if(type=="balance_sheet"){var r=ff(cofa=this.cofa["Balance Sheet"],data=data,spc="",al=al_,am)}
  else if (type=="income_statement"){
   var r=ff(cofa=this.cofa["Income"],data=data,spc="",al=ai_,am)}
  return r;
}

AccountingObj.prototype.get_financial_statement = function(statement,address)
{
 //alert(JSON.stringify(this.accounts))
 var fa=function(dic, data_, accounts, ad, sta, ss="")
 {
 var s_='';
 for(a in dic)
 {
  if(!isNaN(parseInt(a)))
  {
   s_+='<tr>'+'<td>&nbsp;&nbsp;'+ss+accounts[a]+'</td>';
      for(i in data_)
      {
       s_+='<td>'+data_[i][a]+'</td>'
      };
      s_+='</tr>';
  }
  else
  {if(a!="v")
   {ad1={};s_+='<tr>'+'<td>'+ss+a+'</td>';
    for(i in ad)
    {ad1[i]={};ad1[i][sta]={};ad1[i][sta]=ad[i][sta][a];
     var av="";if(a.includes("Total")){av=100*ad[i][sta][a]["v"];av=Math.round(av)/100};
     s_+='<td>'+av+'</td>'
    }
    s_+='</tr>';s_+=fa(dic[a], data_, accounts, ad1, sta, ss+"&nbsp;&nbsp;")
   }
  }
 }
 return s_
}

 var s='<div>'+address+'</div>';
 s+="<table class=''>"
 s+='<thead><tr><th>Accounts</th>';
 for(i in this.data_for_report){s+='<th>'+i+'</th>'};
 s+='</tr></thead>'
 s+="<tbody>";

 s+=fa(this.data_for_report[2022][statement],this.accounting_data, this.accounts , this.data_for_report, statement)

//s+=fa(this.data_for_report[2022]["Income"], this.accounting_data, this.data_for_report, "Income")
 s+="</tbody></table>"
 return s;
}


// -- acObj --
function acObj(){}

acObj.prototype.create_obj = function(){
  var container = document.getElementById("content_"+this.data["container_id"])
  if(container==null){var container = document.getElementById(this.data["container_id"])}
  //alert("9015"+JSON.stringify(this.data));
  //alert("9016"+container.outerHTML)
  //alert(container) //alert(container.outerHTML)
  var ee_=this.data["element_name"];if(ee_=="RichText"){ee_="div"}
  if(ee_=="Object"){ee_="div"}
  this.new_obj=document.createElement(ee_);
  this.new_obj.my_creator_obj=this;
  this.new_obj.is_call_on_change=true;
  var call_on_change=dic["properties"]["call_on_change"];
  if(call_on_change!=null && call_on_change!=""){

  //alert("1\n="+call_on_change+"=\n"+this.data["properties"]["obj_number"])

  this.new_obj.is_call_on_change=call_on_change;}

  this.new_obj.setAttribute("container_id", this.data["container_id"]);
  this.new_obj.setAttribute("id", this.data["properties"]["obj_number"]);
  this.new_obj.setAttribute("obj_type", this.data["obj_type"]);
  this.new_obj.setAttribute("type_", this.data["element_name"]);
  if(this.data["element_name"]!="Textarea")
  {this.new_obj.innerHTML=this.data["properties"]["title"];}
  // -- attribute --
  for (k in this.attributes)
  {
   if(k in this.data["properties"]){this.new_obj.setAttribute(k, this.data["properties"][k])}
   else{if (k != "download"){this.new_obj.setAttribute(k, "")}}
  }
  if("width" in this.data["properties"]){var width_=this.data["properties"]["width"];} else {var width_="100"}
  if("height" in this.data["properties"]){var height_=this.data["properties"]["height"];} else {var height_="100"}
  var u="px";if(width_.includes("%")){u=""};

  var font_weight_=this.data["properties"]["font-weight"];
  if(font_weight_=="" || font_weight_==null){var font_weight="";} else {var font_weight=";font-weight:"+font_weight_}
  var font_size=this.data["properties"]["font-size"];
  if(font_size==null){font_size=""} else {font_size=";font-size:"+font_size+"%"}
  var font_family=this.data["properties"]["font-family"];
  if(font_family==null){font_family=""} else {font_family=";font-family:"+font_family}

  var align_=this.data["properties"]["text-align"];if(align_=="" || align_==null){
    if(this.data["element_name"]=="Button"){var align="center"} else{var align="left"}} else {var align=align_}

  var color_=this.data["properties"]["color"]; if(color_=="" || color_==null){var color="black";} else {var color=color_}
  var background_color_=this.data["properties"]["background-color"];if(background_color_=="" || background_color_==null){var background_color_="";}
  else {var background_color_=";background-color:"+background_color_}

  if(this.data["element_name"]=="Input")
  {
    //alert("9015-15"+JSON.stringify(this.data));
    var s_label=this.data["properties"]["title"]+":&nbsp;";
    s_label_length=7.7*(1*s_label.length-5);
    //if(s_label=="Event Date:&nbsp;")
    //{
    var sw=this.data["properties"]["span_width"];if(sw!=null && sw!="") {s_label_length=1*sw}
    //}
    var nx_=s_label_length;var x=1*this.data["properties"]["x"]+nx_;

    var s_style="position:absolute;left:"+x+"px;top:"+this.data["properties"]["y"]+"px;width:"+width_+"px;text-align:"+align+";";

    this.new_obj.setAttribute("style", s_style);
    var span=document.createElement("span");span.innerHTML=s_label;
    var x = 1*this.data["properties"]["x"];

    var s_="position:absolute;left:"+x+"px;top:"+this.data["properties"]["y"]+"px;"
    s+="text-align:right;display:inline-block;width:"+s_label_length+"px;"
    s_+="color:"+color+background_color_;
    span.setAttribute("style", s_);

    container.appendChild(span);

  } else if(this.data["element_name"]=="Object")
  {
    if(atm.is_show_btns==true){this.new_obj.innerHTML="X"}else{this.new_obj.innerHTML=""}
    var obj=document.createElement("Object")
    this.new_obj.appendChild(obj);
    var param=document.createElement("param")
    param.setAttribute("name", "allowfullscreen");
    param.setAttribute("value", "true");
    obj.appendChild(param);
    obj.setAttribute("type", "text/html");
    obj.setAttribute("data", this.new_obj.getAttribute("data"))
    obj.setAttribute("width", this.new_obj.getAttribute("width"))
    obj.setAttribute("height", this.new_obj.getAttribute("height"))

    var s_="position:absolute;left:"+this.data["properties"]["x"]+"px;top:"+this.data["properties"]["y"]+"px;width:"+width_+u+";text-align:"+align+";";
    s_+="color:"+color+background_color_+font_weight+font_size+font_family;
    this.new_obj.setAttribute("style", s_);

  }  else if (this.data["element_name"]=="DIV") {
    var s_ = this.data["properties"]["overflow"];if(s_!=null || s_!=""){s_="overflow:"+s_+";"}
    s_+="color:"+color+background_color_;
    var ss="position:absolute;left:"+this.data["properties"]["x"]+"px;top:"+this.data["properties"]["y"]+"px;"
    ss+="width:"+width_+u+";"+"height:"+height_+u+";"+s_+";text-align:"+align+";";
    this.new_obj.setAttribute("style", ss);
   //"overflow: scroll;
  } else if (this.data["element_name"]=="RichText") {

    var img_=document.createElement("img");
    var x_ = 1*this.data["properties"]["x"];var y_=this.data["properties"]["y"]-10; var radius=12;
    var s_style="position:absolute;cursor:pointer;left:"+x_+"px;top:"+y_+"px;z-index:999;width:"+radius+"px;height:"+radius+"px";
    img_.setAttribute("style", s_style);
    img_.setAttribute("src", "/static/core/text_editor.jpeg");
    container.appendChild(img_);
    img_.container=container;img_.new_obj=this.new_obj;
    img_.addEventListener("click", (event) => {
      var e=event.target;var new_obj=e.new_obj;
      var ce="";try{ce=new_obj.getAttribute("contenteditable")}catch(er){}
      if(ce==null || ce==""){new_obj.setAttribute("contenteditable", "true");
      var rt=atm.get_rich_text_editor(new_obj);new_obj.container.appendChild(rt)}
    });

    this.new_obj.container=container;
    this.new_obj.addEventListener("dblclick", (event) => {
    var e=event.target;var ce="";try{ce=e.getAttribute("contenteditable")}catch(er){}
      if(ce==null || ce==""){e.setAttribute("contenteditable", "true");var rt=atm.get_rich_text_editor(e);e.container.appendChild(rt)}
    });

    this.new_obj.addEventListener("blur", (event) => {
      var e=event.target;
      //alert("c9001-1 "+e.outerHTML);
      var e_container_id_=e.getAttribute("container_id")
      var container = getEBI("content_"+e_container_id_)
      container.onchange(event);
    });

    var s_="overflow-y: auto;";
    var ss="position:absolute;left:"+this.data["properties"]["x"]+"px;top:"+this.data["properties"]["y"]+"px;"
    ss+="width:"+width_+u+";"+"height:"+height_+u+";"+s_+";text-align:"+align+";border-style:solid;border-width:1px;border-color:blue;";
    this.new_obj.setAttribute("style", ss);
    this.new_obj.setAttribute("rich_text_x", 1*this.data["properties"]["x"]);
    this.new_obj.setAttribute("rich_text_y", 1*this.data["properties"]["y"]-65);

    //alert(this.new_obj.outerHTML)
    //alert(this.new_obj.outerHTML)

  } else {
    var s_="position:absolute;left:"+this.data["properties"]["x"]+"px;top:"+this.data["properties"]["y"]+"px;width:"+width_+u+";text-align:"+align+";";
    s_+="color:"+color+background_color_+font_weight+font_size+font_family;
    this.new_obj.setAttribute("style", s_);
  }

  if(this.data["element_name"]=="Select")
  {
   try{this.get_select_data()} catch(er){}
  }

  container.appendChild(this.new_obj)
 for(f in this.data["functions"]){var s="this.new_obj."+f+"="+this.data["functions"][f];eval(s)}
 //alert(this.new_obj.outerHTML);
 //alert(JSON.stringify(this.obj_properties));
 //this.obj_properties;
 //this.functions;
//  alert("A13\n"+this.new_obj.outerHTML)
}

acObj.prototype.get_select_data = function()
{
  var dic=this.data;
  this.new_obj.innerHTML = "";
  //alert("5087-87 acObj.prototype.get_select_data dic=\n"+JSON.stringify(dic["properties"]));
  var options=dic["properties"]["options"];
  var multiple_=dic["properties"]["multiple"];
  if(!(multiple_==null)){this.new_obj.setAttribute("multiple", "multiple")}
  var fun=function(data, html_obj){
                //alert(html_obj.outerHTML)
                //alert(JSON.stringify(data));
                //alert("5087-87-1 acObj.prototype.get_select_data dic=\n"+JSON.stringify(dic["properties"]));
                html_obj.innerHTML=""
                var option = document.createElement("option");
                option.value = "-1"; option.text = "----";
                html_obj.appendChild(option);
                for(i in data){
                 var l=data[i];
                 var option = document.createElement("option");
                 option.value = l[0];
                 option.text = l[1];
                 html_obj.appendChild(option);
                }
                var call_on_change=""; try{call_on_change=dic["properties"]["call_on_change"]}catch(er){}
                if(call_on_change!=""){
  //alert("5\n="+call_on_change+"=\n"+dic["properties"]["obj_number"])
                  var default_value=""; try{default_value=dic["properties"]["default_value"]}catch(er){}
                  if(default_value!=""){html_obj.value=default_value}
                }
              }
  if(options!="" & options!=null)
  {
    options=options.split(",")
    for(i in options){
     var option = document.createElement("option");
     option.value = options[i];
     option.text = options[i];
     this.new_obj.appendChild(option);
    }
  } else if (dic["properties"]["global_adjective"]!="" & dic["properties"]["global_adjective"]!=null)
  {
   var adjective=dic["properties"]["global_adjective"];
   var index=""; try{index=dic["properties"]["index"]} catch(er){};
   var dic_ = {"obj": "AdvancedTabs", "atm": atm_.my_name, "app": "core", "fun": "get_adjective",
               "params": {"app": "core", "model_name": "AdjectivesValues",
               "manager_name": "adjectives","manager_fun": "adjectives",
               "manager_fun_field": "adjective_title","field_value":adjective,
               "index":index}
              }
      //alert(JSON.stringify(dic_))
      this.atm.get_list(call_back_fun=fun, dic_, this.new_obj);
  } else if (dic["properties"]["app_adjective"]!="" & dic["properties"]["app_adjective"]!=null)
  {
   var adjective=dic["properties"]["app_adjective"];
   var index=""; try{index=dic["properties"]["index"]} catch(er){};
   var dic_ = {"obj": "AdvancedTabs", "atm": atm_.my_name, "app": this.atm.my_app, "fun": "get_adjective",
               "params": {"app": this.atm.my_app, "model_name": "AdjectivesValues",
               "manager_name": "adjectives","manager_fun": "adjectives",
               "manager_fun_field": "adjective_title","field_value":adjective,
               "index":index
               }
              }
      //alert(JSON.stringify(dic_))
      this.atm.get_list(call_back_fun=fun, dic_, this.new_obj);
  } else if(dic["properties"]["data_app"]!="" & dic["properties"]["data_app"]!=null)
  {
    //alert("5089-89 acObj.prototype.get_select_data dic=\n"+JSON.stringify(dic["properties"]));
    var data_app=dic["properties"]["data_app"];
    var data_model=dic["properties"]["data_model"];
    var data_field=dic["properties"]["data_field"];
    var data_filter_field=dic["properties"]["data_filter_field"];
    var data_filter_field_value = dic["properties"]["data_filter_field_value"];
    var data_filter_field_ft = dic["properties"]["data_filter_field_ft"];
    //--
    //alert(data_filter_field)
    if(data_filter_field==null){data_filter_field="";data_filter_field_value="";}
    var dic_ = {"obj": "AdvancedTabs", "atm": atm_.my_name, "app": data_app, "fun": "get_list_from_model",
               "params": {"app": data_app, "model_name": data_model,
                          "manager_name": "","manager_fun": "",
                           "manager_fun_field": "","field_name":data_field,
                           "data_filter_field":data_filter_field,
                           "data_filter_field_value":data_filter_field_value,
                           "data_filter_field_ft":data_filter_field_ft
                         }
              }
    try{var v_=dic["properties"]["default_value"];
    if(v_==null || v_==""){var field_=dic["properties"]["field"];

    //alert("field_= "+field_); alert('atm.'+field_+' = '+eval('atm.'+field_))
    try{v_=eval('atm.'+field_)}catch(er){};}
    this.new_obj.run_eval="html_obj.value="+v_} catch(er){}
    //alert(this.new_obj.run_eval)
    //try{var i_=dic["properties"]["default_index"];this.new_obj.run_eval="html_obj.options["+i_+"].selected=true"} catch(er){}
    this.atm.get_list(call_back_fun=fun, dic_, this.new_obj);
  }
}

acObj.prototype.set_select_data_by_dic = function(dic)
{
 this.new_obj.innerHTML="";
 for(var k in dic){var o=document.createElement("option");o.value=k;o.text=dic[k];this.new_obj.appendChild(o)}
}

acObj.prototype.update_data_by_dic = function(dic)
{
  this.new_obj.innerHTML = "";
  var option = document.createElement("option");
  option.value = "-1"; option.text = "----";
  this.new_obj.appendChild(option);
  for(i in dic){
     var option = document.createElement("option");
     option.value = i;
     option.text = dic[i];
     this.new_obj.appendChild(option);
  }
}


// -- acDash dash board --
function acDashboard(){
}

acDashboard.prototype.create_obj = function(){this.creator.create_obj();}

// -- acBasicCreator --
function acBasicCreator(parent){this.parent=parent;
  this.nav_width=33;
 //alert(JSON.stringify(this.parent.data));
 if(atm.grades==null){
  atm.grades={};atm.data_dic={};atm.is_run_set_obj_structure=false;atm.nlimit=-1;
 }
}

acBasicCreator.prototype.get_board = function(dic_,pp_, style_)
{
     var obj_name="BasicDashboard"+pp_["obj_number"]+dic_["board_name"];
     var obj_number = pp_["obj_number"]
     var s='function '+obj_name+'(dic, obj)'
     s+='{'
     s+=' this.content=document.createElement("div");';
     s+=' obj.main_div.appendChild(this.content);';
     s+=' this.content.my_creator_obj=this;';
     style_+="float:left;background-color:"+dic_["background-color"]+";color:"+dic_["color"]+";display:"+dic_["display"]+";";
     s+=' this.content.setAttribute("style", "'+style_+'");';
     s+='};'; try{eval(s)} catch(er){alert("er9002-221: "+er)}
     s = 'new '+ obj_name+'(dic=dic_, obj=this)'; try{obj_=eval(s)} catch(er){alert(er)}
     return obj_;
}

acBasicCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  var pp_=dic["properties"];
  var ll_boards=["battalion","company","platoon","section"]
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  var dic_objs={}
  for(var h in ll_boards){var n=1*h+1;
   var s='var board'+n+'_color="white";if((pp_["board'+n+'_color"]!=null && pp_["board'+n+'_color"]!="")){board'+n+'_color=pp_["board'+n+'_color"]}';eval(s);
   //--
   dic_objs[1*n]={};
   dic_objs[1*n]["board_name"]=ll_boards[h];dic_objs[1*n]["color"]="black";if(n==1){dic_objs[1*n]["display"]="block";}
   var s='dic_objs['+n+']["background-color"]=board'+n+'_color';eval(s);
  }
  //alert("acRadioCreator 90446-16  "+JSON.stringify(dic_objs));
  //--
  var obj_number = pp_["obj_number"]
  var width_=pp_["width"]; if(width_==null || width_==""){width_=80};
  var height_=pp_["height"]; if(height_==null || height_==""){height_=300};
  var color_="black";if((pp_["color"]!=null && pp_["color"]!="")){color_=pp_["color"]}
  var background_color_="white";if(pp_["background-color"]!=null && pp_["background-color"]!=""){background_color_=pp_["background-color"]}
  //--
  this.main_div=document.createElement("div");
  container.appendChild(this.main_div);
  this.main_div.my_creator_obj=this;
  this.main_div.my_creator_obj.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  this.main_div.setAttribute("container_id", dic["container_id"]);
  //this.main_div.setAttribute("class", "row");
  var style_="position:absolute;left:"+pp_["x"]+"px;top:"+pp_["y"]+"px;width:"+(1*width_+1*this.nav_width)+"px;"
  this.main_div.setAttribute("style", style_);
  // --
     var style_ = "float:left;width:"+width_+"px;height:"+height_+"px;"
     if(pp_["border_style"]!=null && pp_["border_style"]!=""){style_+="border-style:"+pp_["border_style"]+";"}
     if(pp_["border_width"]!=null && pp_["border_width"]!=""){style_+="border-width:"+pp_["border_width"]+"px;"}
     if(pp_["border_color"]!=null && pp_["border_color"]!=""){style_+="border-color:"+pp_["border_color"]+";"}
     if(pp_["border_radius"]!=null && pp_["border_radius"]!=""){style_+="border-radius:"+pp_["border_radius"]+"px;"}

  this.nav=document.createElement("div");this.nav.my_creator_obj=this;
  var style_nav="float:left;width:"+this.nav_width+"px;"
  this.nav.setAttribute("style", style_nav)
  this.main_div.appendChild(this.nav);
  this.board_objs = {}
  this.button_objs = {}
  for (var j in dic_objs){this.board_objs[j]=this.get_board(dic_objs[j], pp_, style_)}
  for (var j in dic_objs)
  {
    var s_='var br=document.createElement("br");this.nav.appendChild(br);'
    s_+='this.b_'+j+'=document.createElement("button");this.b_'+j+'.innerHTML="'+dic_objs[j]["board_name"][0].toUpperCase()+'";';
    s_+='this.b_'+j+'.my_creator_obj=this;';
    s_+='this.b_'+j+'.setAttribute("style", "border-radius: 12px;background-color:#04AA6D;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 2px 2px;cursor: pointer;")'
    eval(s_);
    var ss='this.b_'+j+'.addEventListener("click", function(event){';
    ss+='var e=event.target;e.my_creator_obj.hide_boards('+j+');e.my_creator_obj.board_objs['+j+'].content.style.display="block";';
        ss+='try{var s_obj=get_creator(e.my_creator_obj.synchronize_with);';
        ss+='s_obj.hide_boards('+j+');s_obj.board_objs['+j+'].content.style.display="block";}catch(er){}';
    ss+='});this.nav.appendChild(this.b_'+j+');';
    eval(ss)
    var s='this.button_objs[j]=this.b_'+j;eval(s)
  }

  for(f in dic["functions"]){if(f!="on_entity_click" && f!="on_receive_data"  && f!="on_amount_paint")
  {var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}}
}

acBasicCreator.prototype.set_detail_data = function(ll){

   // alert("set_detail_data=\n"+ll)

  }

acBasicCreator.prototype.hide_boards = function(n_)
{
    for(var j in this.board_objs){this.board_objs[j].content.style.display="none";this.button_objs[j].style.display="none"}
    for(var i=1; i<n_;i++){this.button_objs[i].style.display="block"}
}

acBasicCreator.prototype.get_grade = function(skill_id, unit_id)
{
 ret = 0;
 try{ret=atm.grades[unit_id][skill_id]} catch(er){}
 return ret;
}


// https://plotly.com/javascript/gauge-charts/
// https://plotly.com/javascript/indicator/
// https://plotly.com/javascript/pie-charts/
// https://www.alt-codes.net/smiley_alt_codes.php

acBasicCreator.prototype.set_board_data_old_toBeDeleted_12_15_2023 = function(ll=[])
{
  //alert("set data1\n"+ll.length)
  if(atm.is_run_set_obj_structure==false){return}
  if(ll.length==0){for(var k in atm.data_dic){ll.push(k)}}
   //alert("acBasicCreator.prototype.set_board_data 90446-25 \n"+JSON.stringify(atm.data_dic));
   //alert(ll)
   //--
   var dic_=this.parent.data;var pp_=dic_["properties"];
   this.synchronize_with=pp_["synchronize_with"];
   var type_="individual"; try{type_=pp_["type"]} catch(er){}
   var skills_dic=JSON.parse(pp_["skills"]);var kpis_dic=JSON.parse(pp_["kpis"]);
   this.kpis_sub_titles=JSON.parse(pp_["kpis_sub_titles"]);

   this.fields_skills=skills_dic["l"];this.skills_cols=skills_dic["cols"];
   //["Fitness","Shooting","Professional","Equipment"]
   this.fields_kpis_=kpis_dic["l"];this.kpis_cols=kpis_dic["cols"];

   var width_=pp_["width"]; if(width_==null || width_==""){width_=900};
   var height_=pp_["height"]; if(height_==null || height_==""){height_=300};
   var dic=atm.data_dic;
   var unit_qualification = atm.general.unit_qualification
   var unit_equipment = atm.general.unit_equipment

   //alert("90441-550-4 \n"+JSON.stringify(unit_qualification));
   //alert("90441-550-5 \n"+JSON.stringify(unit_equipment));

   //alert("90441-550-3 \n"+JSON.stringify(atm.general.unit_qualification));
   var z=-1;
   var unit_qualification_dic = {'wet_day':0,'wet_night':0,'raid_day':0, 'raid_night':0}
   var unit_equipment_dic = {'is_equiped':0}

   for(var j in ll){
    //alert(j + "\n"+ll[j])
    z+=1;var dic=dic[ll[z]];

    var unit_qualification=unit_qualification[ll[z]];for(var x in unit_qualification_dic){unit_qualification_dic[x]=unit_qualification[x]}
    var unit_equipment=unit_equipment[ll[z]];for(var x in unit_equipment_dic){unit_equipment_dic[x]=unit_equipment[x]}
    unit_qualification=unit_qualification["data"];
    unit_equipment=unit_equipment["data"];

    var container_=this.board_objs[(1*z+1)].content;
    try{var kpi=dic["kpi"];var board_title=dic["title"];dic=dic["data"];
   } catch(er){alert(er)}}

   if(z<0){try{var container_=this.board_objs[1].content} catch(er){alert(er)}}
   //alert("90441-550-1  "+JSON.stringify(kpi));
   //alert("90441-550-2  "+JSON.stringify(dic));

   try{this.hide_boards((1*z+1)); container_.style.display="block";} catch(er){alert(er)}
   // alert("90441-550  "+JSON.stringify(dic));
   // alert("90441-550-88  "+JSON.stringify(this.fields_kpis_));

   var n_kpi_objs=Object.keys(this.fields_kpis_).length;
   var w_kpi=125;var h_kpi=200; wl_kpi=50;var wr_kpi=50;
   var n_objs=Object.keys(dic).length;
   var wl=50;var wr=50;var w=200;var n__=(width_-wl-wr)/n_objs;if(n__<200){w=n__}
   var wb_kpi=(1*width_-wl_kpi-wr_kpi-n_kpi_objs*w_kpi)/(n_kpi_objs-1);

   if(type_=="unit"){wb_kpi=(1*width_-wl_kpi-wr_kpi-(2+n_kpi_objs)*w_kpi)/(n_kpi_objs+1)}

   var wb=(1*width_-wl-wr-n_objs*w)/(n_objs-1);

   //alert(wb)
   var w_=wl+this.nav_width;var th2_=1*height_/2
   var w_kpi_=wl_kpi+this.nav_width;

   container_.innerHTML="";
   //--
   var title__=document.createElement("h1");title__.innerHTML="<u><b>"+board_title+"</b></u>";
   var top_=15;var style_ = "position:absolute;left:"+(width_/2-50)+"px;top:"+top_+"px;color:blue"
   title__.setAttribute("style", style_);
   container_.appendChild(title__);
   //--
   // add company logo --
   var company_logo_src="/media/training/images/"+pp_["company_logo"]
   var company_logo_style="position:absolute;left:50px;top:10px;width:150px;height:60px;"
   var company_image=document.createElement("img");
   company_image.setAttribute("src", company_logo_src);
   company_image.setAttribute("style", company_logo_style);
   container_.appendChild(company_image);
   // add battalion logo
   var battalion_logo_src="/media/training/images/"+pp_["battalion_logo"]
   var left_=width_-130
   var battalion_logo_style="position:absolute;left:"+left_+"px;top:10px;width:150px;height:60px;"
   var battalion_image=document.createElement("img");

   battalion_image.setAttribute("src", battalion_logo_src);
   battalion_image.setAttribute("style", battalion_logo_style);
   container_.appendChild(battalion_image);
   //--
   //-- chart KPI --
   var n__=n_kpi_objs;if(type_=="unit"){n__=n_kpi_objs+2}
   //alert(n_kpi_objs+"\n"+type_+"\n"+n__)
   //alert(this.kpis_cols)
   for(var j=1; j<=n__;j++)
   {
     var nl_=this.kpis_cols[j-1];
     var div_=document.createElement("div");
     var top_=80
     var style_ = "position:absolute;left:"+w_kpi_+"px;top:"+top_+"px;width:"+w_kpi+"px;height:"+h_kpi+"px;"
     style_+="border-style:solid;border-width:0px;border-color:blue;border-radius:10px;display: flex;align-items: center;justify-content: center;"
     div_.setAttribute("style", style_);

     if(j<3 || (j==3 && type_=="individual") || (nl_==7 && type_=="unit")){
           var s_kpi1=document.createElement("span");s_kpi1.innerHTML=this.kpis_sub_titles[j-1];
           var style_ = "z-index:1;position:absolute;left:"+(w_kpi_+10)+"px;top:"+(top_+150)+"px;color:blue"
           s_kpi1.setAttribute("style", style_);
           container_.appendChild(s_kpi1);
       //--
       var kpi_pass=atm.general.pass_grade["kpi_"+type_][nl_];
       //alert(j+"\n"+kpi_pass+"\n"+JSON.stringify(kpi[nl_])+"\n"+type_+"_grade"+"\n"+kpi[nl_][type_+"_grade"])

       var suffix_="";if(type_=="unit" || (j==3 && type_=="individual")){suffix_="%"}

       var data = [
          {
            type: "indicator",
            mode: "gauge+number",  //+delta
            value: kpi[nl_][type_+"_grade"], number: { suffix: suffix_ },
            delta: { reference: 0, increasing: { color: "RebeccaPurple" } },
            gauge: {
              axis: { range: [null, 100], showticklabels:false, tickwidth: 1, tickcolor: "darkblue" },
              bar: { color: "lightyellow" },
              bgcolor: "lightyellow",
              borderwidth: 2,
              bordercolor: "gray",
              steps: [{ range: [0, kpi_pass], color: "red" },{ range: [kpi_pass, 100], color: "green" }],
              threshold: {line: { color: "blue", width: 4 },thickness: 0.5, value: kpi_pass}
            }
          }
        ];
        //alert(type_+"\n"+j+"\n"+kpi_pass+"\n"+JSON.stringify(data))
        var layout = {
          autosize: false,
          title: { text: this.fields_kpis_[j-1], font: { size: 15 } },
          width: 180,height: 160,
          margin: {l: 5, r: 5, b: 10, t: 20,  pad: 1 },
          paper_bgcolor: container_.style.backgroundColor,
          showlegend: false,
          font: { color: "darkblue", family: "Arial" }
        };
        Plotly.newPlot(div_, data, layout, {displayModeBar: false});
    } else {
        //alert("j="+j+"\ntype="+type_+"\nnobjs="+n_kpi_objs+"\nn__="+n__+"\nnl_="+nl_)
       if(type_=="unit" && j >=n__-1) {
           if(j==n__-1){
               var ll_ = []; var is_all=1
               for(var x in unit_qualification_dic)
               {
                if(unit_qualification_dic[x]==0){ll_.push(x);}else{ll_.push("")}
                is_all*=unit_qualification_dic[x];
               }
               if(is_all==1){var c_='rgb(0, 128, 0)';}else{var c_='rgb(255, 0, 0)'}
               var c=[c_,c_,c_,c_];

                  var data = [{'labels': ll_, 'marker': {'colors': c},
                  text: ll_,
                  textinfo: 'text',
                    'type': 'pie','hoverinfo': 'label','sort': false,
                  }]
                  var layout = {
                      title: { text: "Unit Training", font: { size: 15 } },
                      width: 150,height: 160,
                      margin: {l: 5, r: 5, b: 10, t: 20,  pad: 1 },
                      paper_bgcolor: container_.style.backgroundColor,
                      showlegend: false,
                      hovermode: false,
                      font: { color: "darkblue", family: "Arial" }
                    };
            }
           else if(j==n__)
           {
             //alert("KKKKKKKKKKKK")
               var ll_ = [];var is_all_e=1
               for(var x in unit_equipment_dic)
               {
                if(unit_equipment_dic[x]==0){ll_.push(x);}else{ll_.push("")}
                is_all_e*=unit_equipment_dic[x];
               }
               if(is_all_e==1){var c_='rgb(0, 128, 0)';}else{var c_='rgb(255, 0, 0)'}
               var c=[c_];
                  var data = [{'labels': ll_, 'marker': {'colors': c},
                  text: ll_,
                  textinfo: 'text',
                    'type': 'pie','hoverinfo': 'label','sort': false,
                  }]
                  var layout = {
                      title: { text: "Unit Equipment", font: { size: 15 } },
                      width: 150,height: 160,
                      margin: {l: 5, r: 5, b: 10, t: 20,  pad: 1 },
                      paper_bgcolor: container_.style.backgroundColor,
                      showlegend: false,
                      hovermode: false,
                      font: { color: "darkblue", family: "Arial" }
                    };
           }
        Plotly.newPlot(div_, data, layout, {displayModeBar: false});
       } else {
//            var bgcolor = "#CCD1D1";
//            var v_ = kpi[nl_][type_+"_ref1"];var reference_=kpi[nl_][type_+"_grade"];
//            var data = [
//              {
//                type: "indicator",
//                mode: "number+delta",
//                value: v_,
//                number: { prefix: "" },
//                delta: { position: "top", reference: reference_ }
//              }
//            ];
//            var layout = {
//                  autosize: true,
//                  title: { text: this.fields_kpis_[j-1], font: { size: 20 } },
//                  width: 150,height: 130,
//                  margin: {l: 5, r: 5, b: 10, t: 50,  pad: 1 },
//                  paper_bgcolor: bgcolor,
//                  showlegend: false,
//                  font: { color: "darkblue", family: "Arial" }
//            };
//        Plotly.newPlot(div_, data, layout, {displayModeBar: false});

            var www_=1*div_.style.width.replace('px', '')+10

            div_.style.width=www_+"px"
            //alert(div_.style.width)
            var e_div = document.createElement("div");div_.appendChild(e_div);
            style_="border-radius: 20px;text-align: center; z-index: 1;border: 2px outset red;background-color: lightblue; width:"+(w_kpi+10)+"px; height:"+(h_kpi-50)+"px"
            e_div.setAttribute("style", style_)

     //var style_ = "position:absolute;left:"+w_kpi_+"px;top:"+top_+"px;width:"+w_kpi+"px;height:"+h_kpi+"px;"
            var h4_ = document.createElement("h4");e_div.appendChild(h4_);
            h4_.setAttribute("style", "color:blue; margin-top:5px;");
            h4_.innerHTML= this.fields_kpis_[j-1];

            var v_ = kpi[nl_][type_+"_ref1"];
            var value = document.createElement("p");e_div.appendChild(value);
            value.setAttribute("style", "position:absolute;left:30px;top:95px;color:blue; font-size:35px")
            value.innerHTML = v_

            var reference_=kpi[nl_][type_+"_grade"];
            var p_ = document.createElement("p");e_div.appendChild(p_);
            p_.setAttribute("style", "position:absolute;left:30px;top:50px;color:green; font-size:35px")
            p_.innerHTML = 1*v_ - 1*reference_

            //alert(nl_+"\n\nHH kpi[nl_]\n\n"+type_+"_ref1"+"\n\n"+v_+"\n\n"+JSON.stringify(kpi[nl_]));
            //alert(v_+"\n"+reference_)

        }
    }

     //-- Grade --
     w_kpi_=w_kpi_+1*(1*w_kpi+1*wb_kpi);
     container_.appendChild(div_);
   }
   //--
//         alert("AA unit_qualification\n"+JSON.stringify(unit_qualification));
//         alert("AA unit_equipment\n"+JSON.stringify(unit_equipment));
   var k = 0
   //alert("90441-350-35  "+JSON.stringify(dic));
   if(ll.length<(atm.nlimit+1))
   {
    var n=0;
    for(var z in dic){
      n+=1
        //alert("z "+JSON.stringify(dic[z]["kpi"]));
        var labels_=[]
        var data_=[]
        var background_color_=[]
        for(var j in this.kpis_cols)
        {
         var k_=this.kpis_cols[j]

         var bb_ = this.fields_kpis_[j].replace('<br>', ' ')

         labels_.push(bb_)
         var nn__=1*dic[z]["kpi"][k_][type_+"_grade"]
         if((dic[z]["kpi"][k_][type_+"_grade"]+"")=="NaN"){nn__=0}

         if([1,2].includes(1*k_)==false && k_!=7 && (k_!=4)){
            var ref1=1*dic[z]["kpi"][k_][type_+"_ref1"];
            var grade=dic[z]["kpi"][k_][type_+"_grade"];
            nn__=100*(ref1-grade)/ref1;
             //alert("j="+j+"\nk_="+k_+"\nref1="+ref1+"\ngrade="+grade+"\nnn__"+nn__)
         }
         data_.push(nn__)
         // alert("j="+j+"\nk_="+k_+"\ndata_= "+data_)
         if(nn__>=1*dic[z]["kpi"][k_][type_+"_pass"]){var g='rgb(0,128,0)'}else{var g='rgb(255, 0, 0)'}
         background_color_.push(g)
        }

         var text_color="black"

         var is_all=1;var is_all_e=1;
         var uq=unit_qualification[z];
         var ue=unit_equipment[z];

         //alert(z+"\nuq\n"+JSON.stringify(uq));
         //alert(z+"\nue\n"+JSON.stringify(ue));

         for(var x in unit_qualification_dic){is_all*=uq[x]}
         for(var x in unit_equipment_dic){is_all_e*=ue[x]}

         var text_color="black";var text_bgcolor="red";if(is_all==1 && is_all_e==1){text_color="green";var text_bgcolor="none"}
         //var text_color_e="black";var text_bgcolor_e="red";if(is_all_e==1){text_color_e="green";var text_bgcolor_e="none"}

      var div_=document.createElement("div");
      var div_c=document.createElement("div");
      div_.appendChild(div_c);
      var canvas_=document.createElement("canvas");
      div_.appendChild(canvas_);
      container_.appendChild(div_);

      var style_ = "position:absolute;left:"+w_+"px;top:"+(th2_+20)+"px;width:"+w+"px"
      div_.setAttribute("style", style_);
      w_+=wb+w

      if(type_=="unit"){
      var style_c="background-color:"+text_bgcolor+";font-weight: bold;color:"+text_color+";width:40%;height:30%;position:absolute;top:45%;left:0;margin-top:-15px;line-height:19px;"
      style_c+="border-radius:45%;margin-left:30%;padding:10px;;text-align:center;z-index:0"
      } else{
      var style_c=";font-weight: bold;width:100%;height:30%;position:absolute;top:45%;left:0;margin-top:-15px;line-height:19px;"
      style_c+="padding:10px;;text-align:center;z-index:0"
      }
      div_c.setAttribute("style", style_c);
      div_c.innerHTML=dic[z]["title"];
      //alert(type_+"\n"+div_c.outerHTML)
      var ll_=clone_dic(ll);ll_.push(1*z);div_.ll=ll_;div_.obj=this;
      div_.addEventListener("click", function(event){

       var e=event.target;
        this.obj.set_board_data(ll=this.ll, obj=this.obj);
        //--
        try{var s_obj=get_creator(this.obj.synchronize_with);s_obj.set_board_data(ll=this.ll, obj=s_obj);}catch(er){}
        //--
       if(this.ll.length>2){this.obj.set_detail_data(ll=this.ll)}

      })
      const data = {
        labels: labels_,
        datasets: [{data: data_,backgroundColor: background_color_,hoverOffset: 4}]
      };
      const options = {plugins: {legend: {display: false}}}
      const config = {type: 'doughnut',data: data, options:options};
      var c=new Chart(canvas_, config);
   }
   } else {
      k +=1
      var col_width=125;
      var soldier_col_width=300

      //alert(this.fields_skills)

      var n_cols=this.fields_skills.length
      var n_left=(1*width_-n_cols*col_width-soldier_col_width-this.nav_width)/2+this.nav_width;

      color__="lightblue"
      var div_=document.createElement("div");
      var style_ = "position:absolute;left:"+n_left+"px;top:"+(th2_+20)+"px;height:"+(height_/2.5)+"px;"
      style_+="border-style:solid;border-width:0px;border-color:blue;display: flex;"
      style_+="background-color:"+color__+";align-items: center;justify-content: center;"
      div_.setAttribute("style", style_);

      div_.ll=ll_;div_.obj=this;
      div_.addEventListener("click", function(event){
       var e=event.target;var dic=this.obj.parent.data;
       try{var s="var zz="+dic["functions"]["on_entity_click"];eval(s);zz(event);} catch(er){}
        //alert(e.outerHTML)
        //alert("90441-750  "+JSON.stringify(this.ll));
      })

      var s='s_data=atm.units_structure';for(var w=0;w<ll.length;w++){s+='[ll['+w+']]["data"]'};eval(s);
      //alert("90441-750  \n"+JSON.stringify(s_data));

      var r_="10"
      var s="<table class='basic' style='display: block;height:"+(Math.round(height_/2.5))+"px;overflow-y: auto;'><tr>"
      s+="<th style='width:"+soldier_col_width+"px;border-top-left-radius:"+r_+"px'>Name</th>"
      for(var i in this.fields_skills){
      var s_="width:"+col_width+"px";
      if(i==(this.fields_skills.length-1)){s_+=";border-top-right-radius:"+r_+"px"}
      s+="<th style="+s_+">"+this.fields_skills[i]+"</th>";
      }

      //s+="<th style='width:"+col_width+"px'>Fit for Duty</th>";
      s+="</tr>"
      s+="<tbody >"
      for(var s_ in s_data){
       s+="<tr><td soldier_id='"+s_+"' style='width:"+soldier_col_width+"px'>"+s_data[s_]["title"]+"</td>"

       for(var i in this.fields_skills)
       {
        var nl_=this.skills_cols[i];
        //alert(nl_);

        var grade=this.get_grade(skill_id=nl_, unit_id=s_);
        var color="white";var background_color = "green"; var img_="&#128512;"

        if(grade==0){color="black";background_color = "white"; var img_=""}
        else if(grade<atm.general.pass_grade[type_][nl_])
        {
          background_color="red"; color="white";var img_="&#128557";
        }
        if(grade<100){grade="&nbsp;&nbsp;"+grade}
        //alert(grade + "\n"  + nl_ + "\n" + atm.general.pass_grade[type_][nl_] + "\n" + background_color)
        s+="<td style='text-align:center;width:100px;color:"+color+";background-color:"+background_color+"'>"+grade+"&nbsp;&nbsp;&nbsp;&nbsp;"+img_+"</td>"
       }
       s+="</tr>"
      }
      s+="</tbody></table>"
      div_.innerHTML=s;
      container_.appendChild(div_);
   }
}

acBasicCreator.prototype.set_board_data = function(ll=[])
{
  //alert("set data1\n"+ll.length)
  if(atm.is_run_set_obj_structure==false){return}
  if(ll.length==0){for(var k in atm.data_dic){ll.push(k)}}
   //alert("acBasicCreator.prototype.set_board_data 90446-25 \n"+JSON.stringify(atm.data_dic));
   //alert(ll)
   //--
   var dic_=this.parent.data;var pp_=dic_["properties"];
   this.synchronize_with=pp_["synchronize_with"];
   var type_="individual"; try{type_=pp_["type"]} catch(er){}
   var skills_dic=JSON.parse(pp_["skills"]);var kpis_dic=JSON.parse(pp_["kpis"]);
   this.kpis_sub_titles=JSON.parse(pp_["kpis_sub_titles"]);
   this.kpis_images=JSON.parse(pp_["kpis_images"]);
   this.kpis_progress_co=JSON.parse(pp_["kpis_progress_co"]);

   this.fields_skills=skills_dic["l"];this.skills_cols=skills_dic["cols"];
   //["Fitness","Shooting","Professional","Equipment"]
   this.fields_kpis_=kpis_dic["l"];this.kpis_cols=kpis_dic["cols"];

   var width_=pp_["width"]; if(width_==null || width_==""){width_=900};
   var height_=pp_["height"]; if(height_==null || height_==""){height_=300};
   var dic=atm.data_dic;
   var unit_qualification = atm.general.unit_qualification
   var unit_equipment = atm.general.unit_equipment

   //alert("90441-550-4 \n"+JSON.stringify(unit_qualification));
   //alert("90441-550-5 \n"+JSON.stringify(unit_equipment));

   //alert("90441-550-3 \n"+JSON.stringify(atm.general.unit_qualification));
   var z=-1;
   var unit_qualification_dic = {'wet_day':0,'wet_night':0,'raid_day':0, 'raid_night':0}
   var unit_equipment_dic = {'is_equiped':0}
   for(var j in ll){
    //alert(j + "\n"+ll[j])
    z+=1;var dic=dic[ll[z]];

    var unit_qualification=unit_qualification[ll[z]];for(var x in unit_qualification_dic){unit_qualification_dic[x]=unit_qualification[x]}
    var unit_equipment=unit_equipment[ll[z]];for(var x in unit_equipment_dic){unit_equipment_dic[x]=unit_equipment[x]}
    unit_qualification=unit_qualification["data"];
    unit_equipment=unit_equipment["data"];

    var container_=this.board_objs[(1*z+1)].content;
//   alert(container_.outerHTML)
    try{var kpi=dic["kpi"];var board_title=dic["title"];dic=dic["data"];
   } catch(er){alert(er)}}

   if(z<0){try{var container_=this.board_objs[1].content} catch(er){alert(er)}}
   //alert("90441-550-1  "+JSON.stringify(kpi));
   //alert("90441-550-2  "+JSON.stringify(dic));
   try{this.hide_boards((1*z+1)); container_.style.display="block";} catch(er){alert(er)}
   // alert("90441-550  "+JSON.stringify(dic));
   // alert("90441-550-88  "+JSON.stringify(this.fields_kpis_));

   var n_kpi_objs=Object.keys(this.fields_kpis_).length;
   var w_kpi=125;var h_kpi=200; wl_kpi=50;var wr_kpi=50;
   var n_objs=Object.keys(dic).length;
   var wl=50;var wr=50;var w=200;var n_=(width-wl-wr)/n_objs;if(n_<200){w=n_}
   var wb_kpi=(1*width_-wl_kpi-wr_kpi-n_kpi_objs*w_kpi)/(n_kpi_objs-1);

   if(type_=="unit"){wb_kpi=(1*width_-wl_kpi-wr_kpi-(2+n_kpi_objs)*w_kpi)/(n_kpi_objs+1)}

   var wb=(1*width_-wl-wr-n_objs*w)/(n_objs-1);

   //alert(wb)
   var w_=wl+this.nav_width;var th2_=1*height_/2
   var w_kpi_=wl_kpi+this.nav_width;
   container_.innerHTML="";
   var div_block = document.createElement("div")
   div_block.setAttribute("style", "position:absolute;left:"+left_+"px;")
   container_.appendChild(div_block)

   var left_  = width_/5
   var div_cycle_u = document.createElement("div")
   div_cycle_u.setAttribute("style", "position:absolute;left:"+left_+"px;")
   div_block.appendChild(div_cycle_u)

  // alert(pp_["height"])

   var hh_ = pp_["height"]/2.7
//alert("hh_ "+hh_)
   var div_canvas_square_u = document.createElement("div")
   div_canvas_square_u.setAttribute("style", "position:absolute;left:"+(-left_-300)+"px;bottom:"+hh_+"px;top:"+hh_+"px")
   div_block.appendChild(div_canvas_square_u)
   //alert(div_block.outerHTML)
   //--
    var board_title_= board_title.split(" ")
    var board_title_1="";if(board_title_[1]!=null && board_title_[1]!=""){board_title_1=board_title_[1]}

   var title__=document.createElement("h1");
   title__.innerHTML="<span style='color:black' ><b>"+board_title_[0]+"</b></span>"+"<span style='color:#AFE1AF'><b>"+board_title_1+"</b></span>";
   var top_=5;
   var style_ = "position:absolute;font-size:35px;text-transform:uppercase;left:"+(width_/2-50)+"px;top:"+top_+"px;color:blue"
   title__.setAttribute("style", style_);
   container_.appendChild(title__);
   //--
   //-- chart KPI --
   var n__=n_kpi_objs;if(type_=="unit"){n__=n_kpi_objs+2}
   //alert(n_kpi_objs+"\n"+type_+"\n"+n__)
   //alert(this.kpis_cols)

//   var progress_color = ["", "#326FB9", "#7ED0B5", "#26A269"]

   var bg_color = ["#C0BFBC"]
   var inner_circle_color = ["#FFFFFF"]
   for(var j=1; j<=n__;j++)
   {
     var nl_=this.kpis_cols[j-1];
//     kpi[j]["data-progress-color"] = this.kpis_progress_co[j-1]
     kpi[j]['data-bg-color'] = bg_color[0]
     kpi[j]['data-inner-circle-color'] = inner_circle_color[0]
     var div_=document.createElement("div");
     var top_=80
    // alert(w_kpi_)
     var style_ = "position:absolute;left:"+w_kpi_+"px;top:"+top_+"px;width:"+w_kpi+"px;height:"+h_kpi+"px;"
     style_+="border-style:solid;border-width:0px;border-color:blue;border-radius:10px;display:flex;align-items: center;justify-content: center;"
     div_.setAttribute("style", style_);
     if(j<3 || (j==3 && type_=="individual") || (nl_==7 && type_=="unit")){
//           var s_kpi1=document.createElement("span");s_kpi1.innerHTML=this.kpis_sub_titles[j-1];
//           var style_ = "z-index:1;position:absolute;left:"+(w_kpi_+10)+"px;top:"+(top_+150)+"px;color:blue"
//           s_kpi1.setAttribute("style", style_);
//           container_.appendChild(s_kpi1);
       //--
       var kpi_pass=atm.general.pass_grade["kpi_"+type_][nl_];
       //alert(j+"\n"+kpi_pass+"\n"+JSON.stringify(kpi[nl_])+"\n"+type_+"grade"+"\n"+kpi[nl][type_+"_grade"])

       var suffix_="";if(type_=="unit" || (j==3 && type_=="individual")){suffix_="%"}

      var grade_value = kpi[nl_][type_+"_grade"]
      var data_bg_color = kpi[j]["data-bg-color"]
      var data_inner_circle_color = kpi[j]["data-inner-circle-color"]

      var img_ = this.kpis_images[j-1]
      var outer_div = document.createElement('div'); div_.appendChild(outer_div)
      var title =  document.createElement('h4'); outer_div.appendChild(title)
      title_style = "position: absolute;top:-30px;font-family: serif;font-size:14px;"
      title.setAttribute('style', title_style)
      title.innerHTML = this.fields_kpis_[j-1]
      outer_div_style = "position: absolute;border-radius: 50%;display:flex;flex-direction: column;justify-content:center;align-items: center;"
      outer_div_style +="box-shadow: 5px 5px 5px grey;width:180px;height:180px;"
      outer_div.setAttribute('style', outer_div_style)
      outer_div.setAttribute('class', 'circular-progress')
      outer_div.setAttribute('data-inner-circle-color', data_inner_circle_color)
      outer_div.setAttribute('data-percentage', grade_value)
      outer_div.setAttribute('data-progress-color', this.kpis_progress_co[j-1])
      outer_div.setAttribute('data-bg-color', data_bg_color)
      var inner_div = document.createElement('div'); outer_div.appendChild(inner_div)
      inner_div_style = "position: absolute;width:150px;height:150px;border-radius: 50%;background-color: lightgrey;"
      inner_div.setAttribute('style', inner_div_style)
      inner_div.setAttribute('class', 'inner-circle')

      var img = document.createElement('img');outer_div.appendChild(img)
      img.setAttribute("src", "/media/training/images/"+img_)
      img_style="position:absolute;top:10px;width:100px;height:100px;color:"+this.kpis_progress_co[j-1]
      img.setAttribute("style", img_style)

      var p = document.createElement('p');outer_div.appendChild(p)
      p_style = "position:relative;top:20px;font-size: 1.2rem;font-weight:bold;color: rgb(0, 0, 0, 0.8);"
      p.setAttribute('style', p_style)
      p.setAttribute('class', 'percentage')
      p.setAttribute('value-color', 'black')
      p.innerHTML = '0%'
      var h6 = document.createElement('h6');outer_div.appendChild(h6)
      header_style = "position:relative;top:25px"
      h6.setAttribute('style', header_style)
      h6.setAttribute('class', 'header')
      h6.innerHTML = this.kpis_sub_titles[j-1];

    } else {
        //alert("j="+j+"\ntype="+type_+"\nnobjs="+n_kpi_objs+"\nn_="+n+"\nnl="+nl_)
       if(type_=="unit" && j >=n__-1) {
           if(j==n__-1){

               var ll_ = []; var is_all=1
               for(var x in unit_qualification_dic)
               {
                if(unit_qualification_dic[x]==0){ll_.push(x);}else{ll_.push("")}
                is_all*=unit_qualification_dic[x];
               }

               if(is_all==1){var c_='rgb(0, 128, 0)';}else{var c_='rgb(255, 0, 0)'}
               var c=[c_,c_,c_,c_];

                var div_shape_container = document.createElement('div');div_.appendChild(div_shape_container)
                var style_ = 'position:absolute;width:180px;height:180px;background-color:#63C0A1;'
                style_+='display:flex;justify-content: center;align-items: center;'
                div_shape_container.setAttribute('style',style_)
                div_shape_container.setAttribute('class', 'shape-container')
                var div_circle = document.createElement('div')
                div_circle.setAttribute('class','top-circle');div_shape_container.appendChild(div_circle)
                var style_ = 'position:absolute;width: 80px;height:80px;background-color:#A7151F;border-radius:50%;top:-25%;left: 50%;border-left: 6px solid white;'
                style_+='border-right: 6px solid white;border-bottom: 6px solid white; transform: translateX(-50%);'
                div_circle.setAttribute('style',style_)
                var div_sharp_edge = document.createElement('div')
                var style_ = 'position:absolute;width: 0;height: 0;border-left: 20px solid transparent;'
                style_+='border-right: 20px solid transparent;border-top: 30px solid #63C0A1;top: 100%;left: 50%;transform: translateX(-50%);'
                div_sharp_edge.setAttribute('style', style_)
                div_sharp_edge.setAttribute('class','sharp-edge');div_shape_container.appendChild(div_sharp_edge)
                 var div_center_text = document.createElement('div')
                div_center_text.setAttribute('class','center-text');div_shape_container.appendChild(div_center_text)
                div_center_text.setAttribute('style','color: #fff;font-size: 16px; ')
                var h5 = document.createElement('h5')
                h5.setAttribute('style', 'text-decoration:underline');div_center_text.appendChild(h5)
                h5.innerHTML = "UNIT TRAINING"

            }
           else if(j==n__)
           {
               var ll_ = [];var is_all_e=1
               for(var x in unit_equipment_dic)
               {
                if(unit_equipment_dic[x]==0){ll_.push(x);}else{ll_.push("")}
                is_all_e*=unit_equipment_dic[x];
               }
               if(is_all_e==1){var c_='rgb(0, 128, 0)';}else{var c_='rgb(255, 0, 0)'}
               var c=[c_];

               var div_shape_container = document.createElement('div');div_.appendChild(div_shape_container)
                var style_ = 'position:absolute;width:180px; height: 180px;background-color:#43695C;margin-left:30px;'
                style_+='display: flex;justify-content: center;align-items: center;'
                div_shape_container.setAttribute('style',style_)
                div_shape_container.setAttribute('class', 'shape-container')
                var div_circle = document.createElement('div')
                div_circle.setAttribute('class','top-circle');div_shape_container.appendChild(div_circle)
                var style_ = 'position:absolute;width: 80px;height:80px;background-color:#7ED0B5;border-radius:50%;top:-25%;left: 50%;border-left: 6px solid white;'
                style_+='border-right: 6px solid white;border-bottom: 6px solid white; transform: translateX(-50%);'
                div_circle.setAttribute('style',style_)
                var div_sharp_edge = document.createElement('div')
                var style_ = 'position:absolute;width: 0;height: 0;border-left: 20px solid transparent;'
                style_+='border-right: 20px solid transparent;border-top:30px solid #43695C;top:100%;left:50%;transform: translateX(-50%);'
                div_sharp_edge.setAttribute('style', style_)
                div_sharp_edge.setAttribute('class','sharp-edge');div_shape_container.appendChild(div_sharp_edge)
                 var div_center_text = document.createElement('div')
                div_center_text.setAttribute('class','center-text');div_shape_container.appendChild(div_center_text)
                div_center_text.setAttribute('style','color: #fff;font-size: 16px; ')
                var h5 = document.createElement('h5')
                h5.setAttribute('style', 'text-decoration:underline');div_center_text.appendChild(h5)
                h5.innerHTML = "UNIT EQUIPMENT"
           }

       } else {


        var value_ = kpi[nl_][type_+"_ref1"];var reference_=kpi[nl_][type_+"_grade"]
        var kk = 1*value_ - 1*reference_

    var outer_div = document.createElement('div'); div_.appendChild(outer_div)
    outer_div_style = "position:absolute;display:flex;flex-direction:column;justify-content:center;align-items:center;"
    outer_div.setAttribute('style', outer_div_style)
    var canvas_= document.createElement("canvas");outer_div.appendChild(canvas_)
    canvas_.setAttribute("id", "shapesCanvas");
    canvas_.setAttribute("width", "300");
    canvas_.setAttribute("height", "300");
    container_.appendChild(div_);
    var title =  document.createElement('h4'); outer_div.appendChild(title)
    title_style = "position: absolute;top:20px;font-family: serif;font-size:14px;"
    title.setAttribute('style', title_style)
    title.innerHTML = this.fields_kpis_[j-1]



        var canvas = document.getElementById("shapesCanvas");
        var context = canvas_.getContext("2d");
        // Set the center and radius of the hexagon
        var hexagonCenterX = canvas.width / 2;
        var hexagonCenterY = canvas.height / 2;
        var hexagonRadius = 100;
        var hexagonRotation = Math.PI / 6; // Rotation angle in radians
        var hexagonTextOffsetY = 70; // Offset for text positioning
        // Draw the rotated hexagon
        drawHexagon(context, hexagonCenterX, hexagonCenterY, hexagonRadius, hexagonRotation, value_, hexagonTextOffsetY);
        // Set the center and radius of the pentagon
        var pentagonCenterX = hexagonCenterX;
        var pentagonCenterY = hexagonCenterY-15;
        var pentagonRadius = hexagonRadius ; // Adjust size as needed
        var pentagonSideLength = 75;
        var pentagonTextOffsetY = 5; // Offset for text positioning
        // Draw the nested pentagon
        drawPentagon(context, pentagonSideLength, pentagonCenterX, pentagonCenterY, pentagonRadius, kk, pentagonTextOffsetY);

        function drawHexagon(context, x, y, radius, rotation, text, textOffsetY) {
            context.beginPath();
            for (var i = 0; i < 6; i++) {
                var angle = (i * 2 * Math.PI) / 6 + rotation;
                var xPos = x + radius * Math.cos(angle);
                var yPos = y + radius * Math.sin(angle);

                if (i === 0) {
                    context.moveTo(xPos, yPos);
                } else {
                    context.lineTo(xPos, yPos);
                }
            }
            context.closePath();
            context.strokeStyle = "#B5B5B3"; // Stroke color
            context.lineWidth = 2; // Stroke width
            context.fillStyle = "#B5B5B3"; // Fill color
            context.fill();
            context.stroke();

            // Add text inside the hexagon
            context.fillStyle = "#fff"; // Text color
            context.font = "bold 25px Arial"; // Font size and family
            context.textAlign = "center"; // Center the text horizontally
            context.textBaseline = "middle"; // Center the text vertically
            context.fillText(text, x, y + textOffsetY);
        }
        function drawPentagon(context,sideLength, x, y, radius, text, textOffsetY) {
            context.beginPath();
            context.moveTo(x, y - sideLength);
            context.lineTo(x - sideLength, 50 + sideLength * Math.sqrt(2) / 2);
            context.lineTo(x - sideLength / 1, y + sideLength * Math.sqrt(3) / 2);
            context.lineTo(x + sideLength / 1, y + sideLength * Math.sqrt(3) / 2);
            context.lineTo(x + sideLength, 50 + sideLength * Math.sqrt(2) /2);

            context.closePath();
            context.strokeStyle = "#59A963"; // Stroke color
            //context.lineWidth = 2; // Stroke width
            context.fillStyle = "#59A963"; // Fill color
            context.fill();
            context.stroke();

            // Add text inside the pentagon
            context.fillStyle = "white"; // Text color
            context.font = "bold 25px Arial"; // Font size and family
            context.textAlign = "center"; // Center the text horizontally
            context.textBaseline = "middle"; // Center the text vertically
            context.fillText(text, x, y + textOffsetY);
        }

        }
    }
     //-- Grade --
     if(type_=="unit"){w_kpi_=w_kpi_+1*(1*w_kpi+1*(wb_kpi+70));if(j<=3){div_cycle_u.appendChild(div_);}else{div_canvas_square_u.appendChild(div_);}}else{w_kpi_=w_kpi_+1*(1*w_kpi+1*(wb_kpi));container_.appendChild(div_);}
 }
  // alert(div_cycle_u.outerHTML)
  // alert(div_canvas_square_u.outerHTML)
   //--
  const circularProgress = document.querySelectorAll(".circular-progress");
  Array.from(circularProgress).forEach((progressBar) => {
      const progressValue = progressBar.querySelector(".percentage");
      const innerCircle = progressBar.querySelector(".inner-circle");
        let startValue = 30,
        endValue = Number(progressBar.getAttribute("data-percentage")),
        speed =0.1,
        progressColor = progressBar.getAttribute("data-progress-color");
        progressValueColor = progressBar.getAttribute("value-color");

        const progress = setInterval(() => {
            startValue++;
            progressValue.textContent = `${endValue.toFixed(1)}%`;
            progressValue.style.color = `${progressValueColor}`;

            innerCircle.style.backgroundColor = `${progressBar.getAttribute(
              "data-inner-circle-color"
            )}`;

            progressBar.style.background = `conic-gradient(${progressColor} ${
              startValue * 3.6
            }deg,${progressBar.getAttribute("data-bg-color")} 0deg)`;
            if (startValue >= endValue) {
              clearInterval(progress);
            }
          }, speed);
  });
//         alert("AA unit_qualification\n"+JSON.stringify(unit_qualification));
//         alert("AA unit_equipment\n"+JSON.stringify(unit_equipment));
   var k = 0
   //alert("90441-350-35  "+JSON.stringify(dic));
   if(ll.length<(atm.nlimit+1))
   {
    var n=0;
    var nn_=Object.keys(dic).length
    var unit_width_=Math.round((width_-100)/nn_);
    if (unit_width_>330){unit_width_=330}
    for(var z in dic){
        n+=1
        //alert("z\n"+z+"\n"+JSON.stringify(dic[z]["kpi"]));
        var labels_=[]
        var data_=[]
        var background_color_=[]
        for(var j in this.kpis_cols)
        {
         var k_=this.kpis_cols[j]
         var bb_ = this.fields_kpis_[j].replace('<br>', ' ')
         labels_.push(bb_)
         var nn__=1*dic[z]["kpi"][k_][type_+"_grade"]
         if((dic[z]["kpi"][k_][type_+"grade"]+"")=="NaN"){nn__=0}
         if([1,2].includes(1*k_)==false && k_!=7 && (k_!=4)){
            var ref1=1*dic[z]["kpi"][k_][type_+"_ref1"];
            var grade=dic[z]["kpi"][k_][type_+"_grade"];
            nn__=100*(ref1-grade)/ref1;
             //alert("j="+j+"\nk_="+k_+"\nref1="+ref1+"\ngrade="+grade+"\nnn_"+nn_)
         }
         data_.push(nn__)
         // alert("j="+j+"\nk_="+k_+"\ndata_= "+data_)
         if(nn_>=1*dic[z]["kpi"][k_][type_+"_pass"]){var g='rgba(38, 162, 105, 1)'}else{var g='rgba(190, 49, 65, 1)'}
         background_color_.push(g)
        }
      var text_color="black"
      var is_all=1;var is_all_e=1;
      var uq=unit_qualification[z];
      var ue=unit_equipment[z];
     //alert(z+"\nuq\n"+JSON.stringify(uq));
     //alert(z+"\nue\n"+JSON.stringify(ue));
     for(var x in unit_qualification_dic){is_all*=uq[x]}
     for(var x in unit_equipment_dic){is_all_e*=ue[x]}
     var text_color="black";var text_bgcolor="red";if(is_all==1 && is_all_e==1){text_color="green";var text_bgcolor="none"}
     //var text_color_e="black";var text_bgcolor_e="red";if(is_all_e==1){text_color_e="green";var text_bgcolor_e="none"}

      var div_=document.createElement("div")
      var canvas_=document.createElement("canvas");
      canvas_.setAttribute("id", "myChart");
      canvas_.setAttribute("width", unit_width_);
      canvas_.setAttribute("height", "200");
      div_.appendChild(canvas_);
      //var div_c=document.createElement("div");
      //div_.appendChild(div_c);
      container_.appendChild(div_);

      if(type_=="unit"){                                                                        //w = unit_width_
          var style_ = "position:absolute;left:"+(w_+50)+"px;top:"+(pp_["height"]-200)+"px;width:"+unit_width_+"px"
          div_.setAttribute("style", style_);
          w_+=wb+w
      }else{
           var style_ = "position:absolute;left:"+w_+"px;top:"+(th2_+20)+"px;width:"+unit_width_+"px"
           div_.setAttribute("style", style_);
           w_+=wb+w
      }
      if(type_=="unit"){
      var style_c="background-color:"+text_bgcolor+";font-weight: bold;color:"+text_color+";width:40%;height:30%;position:absolute;top:45%;left:0;margin-top:-15px;line-height:19px;"
      style_c+="border-radius:45%;margin-left:30%;padding:10px;;text-align:center;z-index:0"
      } else{
      var style_c=";font-weight: bold;width:100%;height:30%;position:absolute;top:45%;left:0;margin-top:-15px;line-height:19px;"
      style_c+="padding:10px;;text-align:center;z-index:0"
      }
      //div_c.setAttribute("style", style_c);
      //div_c.innerHTML=dic[z]["title"];
      //alert(type_+"\n"+div_c.outerHTML)
      var ll_=clone_dic(ll);ll_.push(1*z);div_.ll=ll_;div_.obj=this;
      div_.addEventListener("click", function(event){
       var e=event.target;
        this.obj.set_board_data(ll=this.ll, obj=this.obj);
        //--
        try{var s_obj=get_creator(this.obj.synchronize_with);s_obj.set_board_data(ll=this.ll, obj=s_obj);}catch(er){}
        //--
       if(this.ll.length>2){this.obj.set_detail_data(ll=this.ll)}

      })

      var theThinGreenLinePlugin = {
        afterDatasetsDraw: function(chartInstance, easing) {
       // The context, needed for drawing.
        var ctx = chartInstance.chart.ctx;
        var centerConfig = chartInstance.chart.config.options.elements.center;
      	var fontStyle = centerConfig.fontStyle || 'Arial';
		var txt = centerConfig.text;
        var color = centerConfig.color || '#000';
        var sidePadding = centerConfig.sidePadding || 20;
        var sidePaddingCalculated = (sidePadding/100) * (chartInstance.chart.innerRadius * 2)
        //Start with a base font of 30px
        ctx.font = "30px " + fontStyle;
        //Get the width of the string and also the width of the element minus 10 to give it 5px side padding
        var stringWidth = ctx.measureText(txt).width;
        var elementWidth = (chartInstance.chart.innerRadius * 2) - sidePaddingCalculated;
        // Find out how much the font can grow in width.
        var widthRatio = elementWidth / stringWidth;
        var newFontSize = Math.floor(30 * widthRatio);
        var elementHeight = (chartInstance.chart.innerRadius * 2);
        // Pick a new font size so it will not be larger than the height of label.
        var fontSizeToUse = 16;
		//Set font settings to draw it correctly.
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        var centerX = ((chartInstance.chartArea.left + chartInstance.chartArea.right) / 2);
        var centerY = ((chartInstance.chartArea.top + chartInstance.chartArea.bottom) - 70); // positioning the text
        ctx.font = fontSizeToUse+"px " + fontStyle;
        ctx.fillStyle = color;
        //Draw text in center
        ctx.fillText(txt, centerX, centerY);

       //ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
       // ctx.shadowBlur = 3;
       // ctx.shadowOffsetX = 5;
       // ctx.shadowOffsetY = 2;

        // The first (and, assuming, only) dataset.
        var dataset = chartInstance.data.datasets[0];
        // The dataset's data length.
        var datasetDataLength = dataset.data.length;
        // The model of the first slice.
        var firstDatumModel = dataset._meta[Object.keys(dataset._meta)[0]].data[0]._model;
        // The model of the last slice.
        var lastDatumModel = dataset._meta[Object.keys(dataset._meta)[0]].data[datasetDataLength - 1]._model;
        // The Thin Green Line's radius (measured from the center of the doughnut).
        // Add a few pixels to make the line extend a bit from the slice's outer radius.
        var lineRadius = firstDatumModel.outerRadius - 1; // increases the length of the line at the bottom of the piechart
        // The first point of The Thin Green Line (for the first slice).
        var firstSliceStartX = firstDatumModel.x + lineRadius * Math.cos(firstDatumModel.startAngle);
        var firstSliceStartY = firstDatumModel.y + lineRadius * Math.sin(firstDatumModel.startAngle);
        // The last point of The Thin Green Line (for the last slice).
        var lastSliceEndX = lastDatumModel.x + lineRadius * Math.cos(lastDatumModel.endAngle);
        var lastSliceEndY = lastDatumModel.y + lineRadius * Math.sin(lastDatumModel.endAngle);
        // Save the context, in order to mess with it. We will restore it later.
        ctx.save();
        // Green
        ctx.strokeStyle = 'grey';
        // Thin
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(firstSliceStartX, firstSliceStartY);
        // Go through the center, so that other doughnuts (e.g. 3/4 doughnuts) will make sense.
        ctx.lineTo(firstDatumModel.x, firstDatumModel.y);
        ctx.lineTo(lastSliceEndX, lastSliceEndY);
        ctx.stroke();
        // Restore the context.
        ctx.restore();
    }
   };

     Chart.pluginService.register(theThinGreenLinePlugin);

     var ctx = document.getElementById("myChart");
      const data = {
        labels: labels_,
        datasets: [{data: data_,backgroundColor: background_color_,hoverOffset: 4}]
      };
     const options = {legend: {display: false},layout: {padding: 1}, rotation: 1 * Math.PI,circumference: 1 * Math.PI,cutoutPercentage: 50,elements:{center: {text: dic[z]["title"],color: 'black',fontStyle: 'Arial',sidePadding: 40,}}}
     const config = {type: 'doughnut',data: data, options:options};
    var myChart = new Chart(canvas_, config);

   }
   } else {


      k +=1
      var col_width=125;
      var soldier_col_width=400

      var n_cols=this.fields_skills.length
      var n_left=(1*width_-n_cols*col_width-soldier_col_width-this.nav_width)/2+this.nav_width;

      color__="lightblue"
      var div_=document.createElement("div");

      var style_ = "position:absolute;left:"+(n_left+50)+"px;top:"+(pp_["height"]-150)+"px;height:"+(height_/2.5)+"px;"
      style_+="border-style:solid;border-width:0px;border-color:blue;display: flex;"
      style_+=";align-items: center;justify-content: center;"
      div_.setAttribute("style", style_);

      div_.ll=ll_;div_.obj=this;
      div_.addEventListener("click", function(event){
       var e=event.target;var dic=this.obj.parent.data;
       try{var s="var zz="+dic["functions"]["on_entity_click"];eval(s);zz(event);} catch(er){}
        //alert(e.outerHTML)
        //alert("90441-750  "+JSON.stringify(this.ll));
      })

      var s='s_data=atm.units_structure';for(var w=0;w<ll.length;w++){s+='[ll['+w+']]["data"]'};eval(s);
      //alert("90441-750  \n"+JSON.stringify(s_data));
      var r_="10"
      var s="<table class='soldier' style='display: block;height:"+(Math.round(height_/2.5))+"px;'>"
      s+="<thead style='display: block;overflow-x: hidden;'><tr style='font-size: 1.25em;font-weight: bold;font-family: Bernard MT Condensed'>"
      s+="<th style='width:"+soldier_col_width+"px;padding: 15px;border-top-left-radius:"+r_+"px'>Name</th>"
      //alert(this.fields_skills)
      for(var i in this.fields_skills){
      var s_="width:"+col_width+"px";
      //alert(this.fields_skills.length)
      if(i==(this.fields_skills.length-1)){s_+=";border-top-right-radius:"+r_+"px"}
      s+="<th style="+s_+">"+this.fields_skills[i]+"</th>";
      }
      //s+="<th style='width:"+col_width+"px'>Fit for Duty</th>";
      s+="</tr></thead>"
      s+="<tbody style='display: block;height:200px;overflow-y: auto;overflow-x: hidden;'>"
      for(var s_ in s_data){var u=s_data[s_]["title"].split(":");
       s+="<tr><td soldier_id='"+s_+"' style='padding:10px;width:"+soldier_col_width+"px'>"
       s+="<span style='font-size:25px;display:inline-block;font-weight:bold;font-family: Bernard MT Condensed'>"
       s+=u[0]+"</span><br><span style='font-size:23px;display:inline-block;font-family: URW DIN Cond, Regular '>"+u[1]+"</span></td>"

       for(var i in this.fields_skills)
       {

        var nl_=this.skills_cols[i];
        //alert(nl_);

        var grade=this.get_grade(skill_id=nl_, unit_id=s_);
        var color="white";
        var background_color = "#4dffc3";
        var img_="<img src='/media/training/images/v.png'"

        if(grade==0){color="black";background_color = "white"; var img_=""}
        else if(grade<atm.general.pass_grade[type_][nl_])
        {
          background_color="#d55d5d"; color="white";
          img_="<img src='/media/training/images/x.png'"
        }
        if(img_!=""){img_+=" width='25' height='25' />"}
        if(grade<100){grade="&nbsp;&nbsp;"+grade}
        if(i==(this.fields_skills.length-1)){col_width=(col_width-16)}
        var s__="<td style='width:"+col_width+"px;color:black;justify-content: center;align-items: center;text-align:center;padding: 10px;background-color:"+background_color+"'>"
        s__+="<div style='font-size:23px;font-weight:bold;font-family:Bernard MT Condensed;display:flex;align-items:center;'>"+grade
        s__+="&nbsp;"+img_+"</div></td>"
        //alert(s__)
        s+=s__
       }
       s+="</tr>"
       //alert(s)
      }
      s+="</tbody></table>"
      div_.innerHTML=s;
      container_.appendChild(div_);

   }
}

acBasicCreator.prototype.set_obj_structure = function(ll_)
{
  //alert("set obj1")
  //alert(atm.is_run_set_obj_structure)
  if(atm.is_run_set_obj_structure==true || atm.general.period==null){return}

  var dic=this.parent.data;var pp_=dic["properties"];
  var qualifications_=eval(pp_["qualifications"]);
  var table_=pp_["table"];
  var setup_dictionary=pp_["setup_dictionary"];var general_entity_name=pp_["entity"];
  var data_fields_=["soldier", "skill", "value"];
  atm.nlimit=atm.general.period[atm.general.current_period]["n_limit"];
  atm.units_structure = eval("this.parent.atm.general."+setup_dictionary);
  atm.entity_list = eval("this.parent.atm.general."+general_entity_name+"_list")
  atm.entity_dic = eval("this.parent.atm.general."+general_entity_name+"_dic")

  // gives the values (in ll) of skill for soldiers of a unit (in dic)
  var ff_=function(dic_, ll, grades, sn_)
  {
     //alert("sn_="+sn_)
      for(var k_ in dic_){
        var title=dic_[k_]["title"]; //alert(k_+"-ff_-"+title+"  :: "+sn_)
        if(!title.includes(":")){
          ff_(dic_[k_]["data"],ll, grades, sn_)}
        else {
            //alert("acRadioCreator 90446-16\n  "+k_+"\n"+JSON.stringify(grades));
            //alert("acRadioCreator 90446-17\n  "+JSON.stringify(grades[""+k_]));
            try{var nnn=grades[""+k_][sn_];ll.push(nnn)} catch(er){}
        }
      }
  }

  // create KPIs ---
  var f=function(dic, data_dic, grades, n=0, n_limit_=-1)
  {
     var n_=n+1;
     var kpi_fields_=["kpi1","kpi2","kpi3","kpi4","kpi5","kpi6","kpi7"]
     for(var k in dic)
     {
        //alert(k+"\n11\n"+dic[k]["title"])
        data_dic[k]={};data_dic[k]["kpi"]={};
        var title=dic[k]["title"];data_dic[k]["title"]=title;if(n_==1){data_dic[k]["type"]="battalion"}
        var nn_=0;
        for(var j in kpi_fields_)
        {
          var ll=[];
          var nn=ff_(dic[k]["data"],ll, grades, sn_=j);
          //alert(ll.length)
          var individual_ref1=0;var unit_ref1=0;
          if([0,1].includes(1*j)==true)
          { //fitness Shooting
            var individual_grade=list_average(ll);
            var kk = ll.filter(_=> _>=atm.general.pass_grade["unit"][1*j])
            var unit_grade=100*kk.length/ll.length;
          } else if([2].includes(1*j)){
            // professional
            var kk = ll.filter(_=> _!=0);var kk1 = kk.filter(_=> _!=-1);var kk_1 = kk.filter(_=> _!=1)
            var unit_grade=kk_1.length; var unit_ref1=kk.length
          } else if([3, 4].includes(1*j)){
            // Equip, Prof.Equip
            var kk = ll.filter(_=> _!=0)
            var kk1 = ll.filter(_=> _>=1)
            //alert(j+"\nll=\n"+ll)
            //alert(j+"\nkk1=\n"+kk1)
            var unit_grade=(kk.length-kk1.length); var unit_ref1=kk.length;
            var individual_grade=100*kk1.length/ll.length;//var individual_ref1=ll.length;
         } else if([5, 6].includes(1*j)){

            // fit for duty
            var kk = ll.filter(_=> _==100);
            //var unit_ref1=ll.length; var unit_grade=kk.length;
            var individual_ref1=ll.length; var individual_grade=ll.length-kk.length;
            //alert(kk.length+"\n"+ll.length+"\n"+ll+"\n"+kk+"\n"+individual_grade)
            var unit_grade=100*kk.length/ll.length;
         }
         nn_+=1;
         data_dic[k]["kpi"][nn_]={"title":"",
                                  "individual_pass":atm.general.pass_grade["individual"][j],
                                  "unit_pass":atm.general.pass_grade["unit"][j],
                                  "individual_grade": individual_grade,
                                  "unit_grade":unit_grade,
                                  "individual_ref1":individual_ref1,
                                  "unit_ref1":unit_ref1
                                  };
         data_dic[k]["kpi"][nn_]["title"]=kpi_fields_[j];
        }
        data_dic[k]["data"]={};
        if(n_<=n_limit_){f(dic[k]["data"], data_dic[k]["data"], grades, n_, n_limit_)}
     }
  }

  // calculate additional variables for soldier (qualifications based on (0-fitness, 1-shooting, 3-equipment
  // 5 for individual and 6 for unit)
  var f_grades=function(grades)
  {var skills_idx=[0,1];var fraction_=100/(skills_idx.length+1);
   for(var s in grades)
   {
      // ZZ1
     grades[s][5]=100;grades[s][6]=100;
     for(var h=0;h<skills_idx.length;h++)
     {var j=skills_idx[h];
       //alert(j+"\n"+skills_idx+"\n"+grades[s][j]+"\n"+atm.general.pass_grade["individual"][j]+"\n"+grades[s])
       if(grades[s][j]<atm.general.pass_grade["individual"][j]){grades[s][5]-=fraction_}
       if(grades[s][j]<atm.general.pass_grade["unit"][j]){grades[s][6]-=fraction_}
     }
     // equipment
     // The conditions should be changed based on the set of critical equipment for soldier or for unit.
     // Need to  add a column in the Excel Equipment file for unit. (add another skil 5 then ....)
     if(grades[s][3]<atm.general.pass_grade["individual"][3]){grades[s][5]-=fraction_}
     if(grades[s][3]<atm.general.pass_grade["unit"][3]){grades[s][6]-=fraction_}
     //-----
     grades[s][5]=Math.round(grades[s][5]);
     grades[s][6]=Math.round(grades[s][6]);
   }
  }
  //alert("acRadioCreator 90446-16  "+JSON.stringify(dic));
  try{
       var fun=function(data, ll){
        //alert("data\n"+JSON.stringify(data));

        //for(j in data){
        // alert(j+"\n"+JSON.stringify(data[j]));
        //}

        var grades=ll[0];var f=ll[1]; var nlimit=ll[2]; var ll_=ll[3];
        try{if(Object.keys(data).length==0 || data["id"].length==0){alert("There is no data for this battlion.");return;}} catch (er) {}
        //alert("90446-77 data \n"+JSON.stringify(data["soldier"]))
        var n_ = atm.entity_dic["id"].length
        // pivot soldier data into lists in atm.grades
        // [0,0,0,0,0,0]
        // i=0 fitness; i=1 shooting; i=2 prof.; i=3 equipment; i=4 and i=5 Fit to duty;
         for(var i=0;i<n_;i++)
         {
          var sid_=1*atm.entity_dic["id"][i];      //alert(sid_)
          if(!(sid_ in grades)){grades[sid_]=[0,0,0,0,0,0,0]}
          var idx=-1;
          try{var idx=data["soldier"].indexOf(sid_)}catch(er){continue}
          while(idx>-1){grades[sid_][data["skill"][idx]]=data["value"][idx];
            try{var idx=data["soldier"].indexOf(sid_, 1*idx+1)} catch(er){}
          }
          //if(!(data["soldier"][i] in grades)){grades[data["soldier"][i]]=[0,0,0,0,0,0,0]}
          //grades[data["soldier"][i]][data["skill"][i]]=data["value"][i]
         }
         f_grades(grades)

         //alert("90449-99-7: "+JSON.stringify(atm.units_structure))
         //alert("2\n"+JSON.stringify(grades));
         //alert("90449-99-79-0: "+JSON.stringify(atm.data_dic));

         // fill atm.data_dic
         // var llc= atm.general.get_platoons_ids();
         f(atm.units_structure, atm.data_dic, grades, 0, n_limit_=atm.nlimit);
         //alert("90449-99-79-1: "+JSON.stringify(atm.data_dic));

         for(var k_ in ll_){
            var obj_=get_creator(ll_[k_])
            atm.is_run_set_obj_structure=true;
            obj_.set_board_data()
          }
        //alert(data["pkf_name"])
      }
       var dic__={"model":table_,"number_of_rows":1000000,"filters":{},
           "fields":data_fields_, "order_by":{"field":"soldier", "direction":"ascending"}
      }
      dic__["filters"]["id"]={"value":atm.general.current_battalion, "foreign_table":"soldier__battalion"}
      //alert("90449-99-13: "+JSON.stringify(dic__));
      this.parent.atm.get_data(call_back_fun=fun, dic__, [atm.grades,f, atm.nlimit, ll_])
  } catch(er){alert("1155-515: "+er)}
}


// -- acReport --
function acReport(){}

acReport.prototype.add_filter = function(f,v,ft){this.filters[f]={"value":v,"foreign_table":ft}}

acReport.prototype.create_obj = function(){
  this.filters={};
  // need to turn the order_by to a dictionary of dictionaries one for each field.
  this.order_by={"field":this.data["properties"]["order_by"],"direction":"ascending"};
  this.get_data_dic={"app":this.data["properties"]["app"],
                     "model":this.data["properties"]["table"],
                     "parent_model": "new", "parent_id":"",
                     "number_of_rows":5000000,
                     "filters":this.filters,
                     "order_by":this.order_by,
                     "fields":["level",
                               this.data["properties"]["horizontal_field"],
                               this.data["properties"]["vertical_field"],
                               this.data["properties"]["value_field"]
                              ]
                    }
  var dic=this.data;var pp_=dic["properties"]
  var container = document.getElementById("content_"+dic["container_id"]);
  this.table_=document.createElement("table");this.table_.my_creator_obj=this;
  container.appendChild(this.table_);
  if(!(pp_["table_class"])){pp_["table_class"]="basic"}
  this.table_.setAttribute("class", pp_["table_class"]);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", pp_["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);

  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in pp_){this.table_.setAttribute(s, pp_[s])}
   else{this.table_.setAttribute(s, "")}}

  for(f in dic["functions"]){if(f!="onclick" && f!="on_receive_data"  && f!="on_amount_paint")
  {var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}

  this.table_.setAttribute("style", "position:absolute;left:"+pp_["x"]+"px;top:"+pp_["y"]+"px");
  this.excel_img_=document.createElement("img");container.appendChild(this.excel_img_);
  this.excel_img_.setAttribute("src", "/media/core/images/Logos-Excel-Copyrighted-icon.png");
  this.excel_img_.setAttribute("alt", "buttonpng");
  this.excel_img_.setAttribute("border", "0");

  this.excel_img_.exportTable=this.table_
  this.excel_img_.setAttribute("style", "position:absolute;cursor:pointer;left:"+(1*pp_["x"]+3)+"px;top:"+(1*pp_["y"]+3)+"px;width:25px;height:25px;");
  this.excel_img_.addEventListener("click", function(event){
    var a = document.createElement('a');
    var dataType = 'data:application/vnd.ms-excel';
    var readTable = this.exportTable.outerHTML.replace(/ /g, '%20');
    a.href = dataType + ', ' + readTable;
    table_name = 'ExcelPivotTable-'+this.exportTable.getAttribute('id')+'.xlsx';
    a.download = table_name;
    a.click();
    event.preventDefault();
    alert("Table: "+table_name+" was downloaded")
  })

  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);

  this.tr_h=document.createElement("tr");this.tr_h.my_creator_obj=this;
  this.tr_h.addEventListener("click", function(event){
   var e=event.target;
  })
  var td = document.createElement("td");
  td.setAttribute("style", "background-color:black; color:white");

  var img_=pp_["image"];if(img_==null || img_==""){img_="default_table_img.png"}
  td.innerHTML="<img id='img_"+pp_["obj_number"]+"' src='/media/training/images/"+img_+"' style='width:300px;height:300x;'>";

  this.tr_h.appendChild(td);

  this.tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(this.tr_h);

  this.tbody=document.createElement("tbody");this.tbody.my_creator_obj=this;
  this.tbody.setAttribute("id", "tbody_"+pp_["obj_number"]);
  this.tbody.setAttribute("style", "display: block;height: "+pp_["height"]+"px;overflow-y: auto;overflow-x: hidden;");
  this.table_.appendChild(this.tbody);
  this.table_.onclick=this.row_col_click;
}

acReport.prototype.set_data = function(type, is_level=true){
  //alert("9087-770-1 acReport.prototype.set_data\n")
  try{eval('var zz='+this.data["functions"]["on_receive_data"])} catch(er){}
  try{eval('var aa='+this.data["functions"]["on_amount_paint"])} catch(er){}
  var report=this;

  var fun = function(data,ll){
     var html_obj = ll[0];var get_data_dic = ll[1];
     //alert("90876-1 data=\n"+JSON.stringify(data));
     //for(k in data){alert("k="+k+"\n"+JSON.stringify(data[k]))}
     //alert(JSON.stringify(get_data_dic));
     //alert(html_obj.outerHTML)
     data["dim_titles"]={}
     try{zz(data)} catch(er){};
     //alert(JSON.stringify(data["amount"]));
     //alert(JSON.stringify(data["dim_titles"]));
     //alert(JSON.stringify(report.get_data_dic));

     var vertical_field_=report.data["properties"]["vertical_field"];
     var horizontal_field_=report.data["properties"]["horizontal_field"];
     var value_field=get_data_dic["fields"][(get_data_dic["fields"].length-1)]
     //alert("k="+vertical_field_+"\n"+JSON.stringify(data[vertical_field_]))
     //alert("h="+horizontal_field_+"\n"+JSON.stringify(data[horizontal_field_]))
     var n_=data[vertical_field_].length;
     var pdata={};var v_=[];var h_=[];
     for(var j=0; j<n_;j++)
     {
              if(!(data[vertical_field_][j] in pdata))
              {pdata[data[vertical_field_][j]]={};v_.push(data[vertical_field_][j])}
              if(!(data[horizontal_field_][j] in pdata[data[vertical_field_][j]]))
              {
                 if(!(h_.includes(data[horizontal_field_][j]))){h_.push(data[horizontal_field_][j])}
              }
              //alert(data[vertical_field_][j]+"\n"+data[horizontal_field_][j]+"\n"+data["amount"][j])
              pdata[data[vertical_field_][j]][data[horizontal_field_][j]]={"value":data[value_field][j],"color":"black"};
              try{aa(v=data[vertical_field_][j],
                     h=data[horizontal_field_][j],
                     a=pdata[data[vertical_field_][j]][data[horizontal_field_][j]],
                     f=get_data_dic["filters"]
                     )} catch(er){}
     }

     //alert("pdata=\n"+JSON.stringify(pdata));
     v_=v_.sort();//h_=h_.sort();
     //alert(JSON.stringify(data["dim_titles"][horizontal_field_]));
     report.axes={"v":v_, "h":h_}
     report.pdata=pdata
     report.raw_data=data
     report.creator.create_html(type)
   }
   //alert(is_level)
  //alert("90876-2\n"+JSON.stringify(this.get_data_dic));
  if(is_level==false){this.get_data_dic["fields"] = this.get_data_dic["fields"].filter(item => item !== "level")}
  //alert("90876-3\n"+JSON.stringify(this.get_data_dic));
  //this.get_data_dic["from_fun"]="acReport.prototype.set_data"
  this.atm.get_data(fun, this.get_data_dic, [this.table_, this.get_data_dic])
}

acReport.prototype.row_col_click = function(event)
{
 var e=event.target;
 //alert(JSON.stringify(this.my_creator_obj.data["functions"]["onclick"]));
 try{eval('var zz='+this.my_creator_obj.data["functions"]["onclick"]);zz(event);
 } catch(er){
 //alert("er9013: "+er)
 }
}

// -- acPivot --
function acPivotCreator(parent){
  this.parent=parent;this.sort_by_col_num=null;this.sort_ascending=true;
  // alert(JSON.stringify(this.parent.data));
}

acPivotCreator.prototype.create_html = function(type=null)
{
  //alert(JSON.stringify(this.parent.axes));
  var p=this.parent;
  var dic=p.data;
  var h_=p.axes["h"];var v_=p.axes["v"]

  var vertical_field_=p.data["properties"]["vertical_field"];
  var horizontal_field_=p.data["properties"]["horizontal_field"];

  //alert(JSON.stringify(v_))
  //alert(JSON.stringify(p.raw_data["dim_titles"][vertical_field_]));
  //alert(JSON.stringify(h_) +"\n"+JSON.stringify(p.raw_data["dim_titles"][horizontal_field_]));
  //alert(p.pdata[v_[0]][h_[5]]["value"])

  var sh="<thead id='thead_"+this.parent.data["properties"]["obj_number"]+"'>"
  sh+="<tr id='tr_h_"+this.parent.data["properties"]["obj_number"]+"' style='cursor:pointer'><th col_num=-1></th>";
  for(var j in h_){sh+="<th col_num="+j+">"+p.raw_data["dim_titles"][horizontal_field_][h_[j]]+"</th>"};
  try{
  if(p.data["properties"]["horizontal_title"]!=null && p.data["properties"]["horizontal_title"]!="")
  {sh+="<th>"+p.data["properties"]["horizontal_title"]+"</th>"}}catch(er){}
  sh+="</tr></thead>";
  //alert(JSON.stringify(this.parent.data["properties"]));
  var s="<table>"+sh+"<tbody id='tbody_"+this.parent.data["properties"]["obj_number"]+"'>"
  if(this.sort_by_col_num!=null)
  {
   try{
      var kk={};var kk_=[]

      if(this.sort_by_col_num == -1)
      {
        for(var h in v_)
        {
         k_idx = p.raw_data["dim_titles"][vertical_field_][v_[h]];
         if(k_idx != null)
         {
           kk[k_idx] = v_[h];kk_.push(k_idx)}
        }
          if(this.sort_ascending==true){kk_=kk_.sort();this.sort_ascending=false}
          else{kk_=kk_.sort().reverse();this.sort_ascending=true}
      } else
      {
          for (var i in v_)
          {
            try{var k_idx=parseFloat(p.pdata[v_[i]][h_[this.sort_by_col_num]]["value"])} catch(er){k_idx = -999999999}
            while (kk_.includes(k_idx)){k_idx=1*k_idx+0.000001}
            kk[k_idx]=v_[i];kk_.push(k_idx);
          }
          if(this.sort_ascending==true){kk_=kk_.sort((a, b) => a - b);this.sort_ascending=false}
          else{kk_=kk_.sort((a, b) => b - a);this.sort_ascending=true}
      }
      //alert(JSON.stringify(kk_));
      var v_=[];for(var g in kk_){v_.push(kk[kk_[g]])}
   } catch(err){alert(err)}
  }

  var ht_="";
  if(dic["properties"]["horizontal_title"]!=null && dic["properties"]["horizontal_title"]!=""){ht_=dic["properties"]["horizontal_title"]}
  var vt_="";
  if(dic["properties"]["vertical_title"]!=null && dic["properties"]["vertical_title"]!=""){vt_=dic["properties"]["vertical_title"]};
  vt_ll=[];for(var j in h_){vt_ll.push(0)};if(vt_!="" && vt_!=""){vt_ll.push(0)};

  for (var i in v_)
  {
      //alert(i+"\n"+v_[i]+"\n"+vertical_field_+"\n")
      //alert(p.raw_data["dim_titles"][vertical_field_][v_[i]])
      s+="<tr><th style='white-space: pre;text-align:left'>"+p.raw_data["dim_titles"][vertical_field_][v_[i]]+"</th>"
      var horizontal_total=0;
      var sss_=""
      for(var j in h_)
      {
       try{
         var s_color="black"; try{s_color=p.pdata[v_[i]][h_[j]]["color"]}catch(er){}
         var ssnn=nice_number(p.pdata[v_[i]][h_[j]]["value"])
         s+="<td style='text-align:right;color:"+s_color+"'>"+ssnn+"</td>"
         sss_+=ssnn
         horizontal_total+=1*p.pdata[v_[i]][h_[j]]["value"]
         vt_ll[j]+=1*p.pdata[v_[i]][h_[j]]["value"]
         } catch(er){s+="<td style='text-align:right'></td>"}
      }
      //alert(i + "  " + JSON.stringify(vt_ll));
      //alert(i + " " +v_[i] + "\n" + sss_)
      //alert(horizontal_total)
      //alert(dic["properties"]["horizontal_title"])
      horizontal_total=Math.round(100*horizontal_total)/100
      try{if(ht_!=""){s+="<td style='text-align:right'>"+horizontal_total+"</td>";
      //alert("j="+ j +"   vt_ll[j+1]=" + vt_ll[1*j+1] + "  i=" + i + "  horizontal_total=" + horizontal_total + " : vt_ll" + JSON.stringify(vt_ll));
      vt_ll[1*j+1]+=1*horizontal_total;
      //alert("j="+ j +"   vt_ll[j+1]=" + vt_ll[1*j+1] + "  i=" + i + "  horizontal_total=" + horizontal_total + " : vt_ll" + JSON.stringify(vt_ll));
      }}catch(er){}
      s+="</tr>"
  }
  var sv="<tr><th style='white-space: pre;text-align:left'>"+vt_+"</th>"
  for(var j in h_)
  {
    sv+="<td style='text-align:right'>"+vt_ll[j]+"</td>"
  }
  if(ht_!=""){
    sv+="<td style='text-align:right'>"+vt_ll[1*j+1]+"</td>"
  }
  sv+="</tr>"
  // --
  //alert("sv\n"+sv)
  if(vt_!=""){s+=sv}

  s+="</tbody></table>";
  //alert("sssss\n"+s)
  p.table_.innerHTML=s;
  p.tbody=document.innerHTML=document.getElementById('tbody_'+this.parent.data["properties"]["obj_number"])
  p.tr_h=document.innerHTML=document.getElementById('tr_h_'+this.parent.data["properties"]["obj_number"])
  p.head=document.innerHTML=document.getElementById('thead_'+this.parent.data["properties"]["obj_number"])
  p.tr_h.my_creator_obj=this;
  p.tr_h.addEventListener("click", function(event){
   var e=event.target;
   //alert(e.outerHTML)
   //alert(e.getAttribute("col_num"))
   this.my_creator_obj.sort_by_col_num=parseInt(e.getAttribute("col_num"));
   this.my_creator_obj.create_html()
  })
}

// -- acFSPivot --
function acFSPivotCreator(parent){
  this.parent=parent;
  // alert(JSON.stringify(this.parent.data));
}

acFSPivotCreator.prototype.create_html = function(type="balance_sheet")
{
  var p=this.parent
  var h_=p.axes["h"];var v_=p.axes["v"]
  var vertical_field_=p.data["properties"]["vertical_field"];
  var horizontal_field_=p.data["properties"]["horizontal_field"];

//alert("p.data\n\n"+JSON.stringify(p.data));
//alert("p.raw_data\n\n"+JSON.stringify(p.raw_data));
//alert("p.raw_data['dim_titles']\n\n"+JSON.stringify(p.raw_data["dim_titles"]));
//alert("p.raw_data['dim_titles']['time_dim']\n\n"+JSON.stringify(p.raw_data["dim_titles"]["time_dim"]));
//alert("p.pdata\n\n"+JSON.stringify(p.pdata));

  var time_dim= p.raw_data["dim_titles"]["time_dim"]
  var r=this.parent.atm.accounting_obj.set_accounting_data(p.pdata, time_dim, type)
  var s="<thead><th>Accounts \\ Month</th>"

  for(var j in h_){s+="<th>"+p.raw_data["dim_titles"][horizontal_field_][h_[j]]+"</th>"};
  try{
  if(p.data["properties"]["horizontal_title"]!=null && p.data["properties"]["horizontal_title"]!="")
  {s+="<th>"+p.data["properties"]["horizontal_title"]+"</th>"}}catch(er){}
  s+="</thead><tboday>"+r[1]+"</tboday>"
  this.parent.table_.innerHTML=s
}

acFSPivotCreator.prototype.row_col_click = function(event)
{
 //var e=event.target;
 //alert(e.outerHTML)
 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event)} catch(er)
 {//alert("er9013: "+er)
 }
}

// -- acgrades --
function acGradesCreator(parent){
  this.parent=parent;
  //try{alert("90000-121  "+JSON.stringify(this.parent.data)) } catch(er){alert(er)}
  this.my_name="grades";this.tests_dic=null;this.group_dic=[];
}

acGradesCreator.prototype.create_html = function(type=null)
{
 alert(2222222);
}

acGradesCreator.prototype.set_obj_structure = function(tests_dic=null, group_dic=null, tests_config=null, entity_config=null)
{
 //alert("90100-1\n"+JSON.stringify(tests_dic)); //alert("90100-2\n"+JSON.stringify(group_dic));
 //alert("90160-3\n"+JSON.stringify(tests_config)); //alert("90160-4\n"+JSON.stringify(entity_config));
 if(tests_dic!=null){this.tests_dic=tests_dic};
 if(group_dic!=null){
  if(group_dic["checked"] == true){this.group_dic = this.group_dic.concat(group_dic["entities"])}
  else {this.parent.tbody.innerHTML=""; this.group_dic = this.group_dic.filter((item) => !group_dic["entities"].includes(item))}
 };

 //alert("90100-11\n"+JSON.stringify(this.tests_dic)); //alert(this.group_dic.length)

 if(this.tests_dic!=null && this.group_dic.length>0){
    this.tests_config=tests_config;this.entity_config=entity_config;
    var dic= {"obj":"TrainingDataProcessing", "atm":atm.my_name, "app":atm.my_app, "obj_param":{"app":atm.my_app, "topic_id":"general"},
              "fun":"get_variable_data", "params":{"app":atm.my_app,"group_dic":this.group_dic,"tests_dic":this.tests_dic}}
    var fun=function(data, this_obj)
    {
      var tests_config=this_obj.tests_config;var group_dic=this_obj.group_dic;
      var var_name=this_obj.tests_dic["var_name"];
      var f=function(v, tc){var n__=tc["from"].length;var r_=-1;
       for(var h=0; h<n__;h++){if(tc["from"][h] <= v && v < tc["to"][h]){r_=tc["grade"][h];break}};return r_;
      }
      var entity_name="soldier";var result=data["result"];var test_dic=data["test_dic"];var var_dic=data["var_dic"]
      var up_value=var_dic["up_value"];var up_color=var_dic["up_color"];var down_value=var_dic["down_value"];
      var down_color=var_dic["down_color"];var other_color=var_dic["other_color"];
      //alert("90160-6\n"+JSON.stringify(result));
      //alert("90160-6\n"+JSON.stringify(result[entity_name+"_number"]));
      //alert("90160-6\n"+JSON.stringify(group_dic));
      //alert("90160-61\n"+JSON.stringify(group_dic["entities"]));
      var n_group=group_dic.length;var n_tests=Object.keys(test_dic).length;
      var userid_width=70;var name_width=125;var event_date_width=100;var col_width=80;
      //alert("90160-7\n"+JSON.stringify(group_dic)+"\n\n"+result[entity_name+"_number"]);

      var ff=function(dic_, idx=0){
          var entity_number=dic_["entity_number"];var entity_name=dic_["entity_name"];var result=dic_["result"];
          var n_tests=dic_["n_tests"];var tests_config=dic_["tests_config"];var test_dic=dic_["test_dic"];
          var e_first_name=dic_["e_first_name"];var e_last_name=dic_["e_last_name"];

          var s="";
          try{var idx=result[entity_name+"_number"].indexOf(entity_number, idx)}catch(er){};
          var entity_test_value={"best":{}, "worst":{}}
          if(idx > -1){
              // in this while, we get all the tests of all events of soldier
              while(idx>-1){
                 var event_id=result["event_id"][idx];var event_date=result["event_date"][idx]
                 if(!(event_id in entity_test_value)){entity_test_value[event_id]={"event_date":event_date}}
                 var value=result["value"][idx];
                 var idx_=tests_config["test_number"].indexOf(""+result["test_number"][idx]);
                 var tc=tests_config["test_grade_conversion"][idx_];var converted_value=f(value,tc);
                 //result["value_converted"].push(converted_value);
                 entity_test_value[event_id][result["test_number"][idx]]={"value":value, "converted_value":converted_value}
                 //--
                 if(!(result["test_number"][idx] in entity_test_value["best"]))
                 {entity_test_value["best"][result["test_number"][idx]]={"value":-100, "converted_value":-100}}
                 if(!(result["test_number"][idx] in entity_test_value["worst"]))
                 {entity_test_value["worst"][result["test_number"][idx]]={"value":1000, "converted_value":100}}
                 //--
                 var etvb_c=entity_test_value["best"][result["test_number"][idx]]["converted_value"]
                 var etvw_c=entity_test_value["worst"][result["test_number"][idx]]["converted_value"]
                 var etve_c=entity_test_value[event_id][result["test_number"][idx]]["converted_value"]
                 if(etvb_c<etve_c){
                  entity_test_value["best"][result["test_number"][idx]]=entity_test_value[event_id][result["test_number"][idx]]
                 }
                 if(etvw_c>etve_c){
                  entity_test_value["worst"][result["test_number"][idx]]=entity_test_value[event_id][result["test_number"][idx]]
                 }
                 try{idx=result[entity_name+"_number"].indexOf(entity_number, idx+1)} catch(er){}
              }
              for(event_id_ in entity_test_value)
              {
                if(event_id_=="best" || event_id_=="worst"){continue}

                  s+="<tr><td style='width:"+userid_width+"px'>"
                  s+=e_userid+"</td>"
                  s+="<td style='width:"+name_width+"px'>"+e_first_name+"</td>";
                  s+="<td style='width:"+name_width+"px'>"+e_last_name+"</td>";
                  s+="<td style='width:"+event_date_width+"px'>"+entity_test_value[event_id_]["event_date"]+"</td>"
                  var var_value=0;var var_weight=0
                  for(var test_number in test_dic)
                  {var etv=entity_test_value[event_id_][""+test_number];
                   if(etv!=null && (1*etv["converted_value"])>-1)
                   {
                    s+="<td v='"+etv["value"]+"' style=';text-align:right;width:"+col_width+"px'>"+etv["converted_value"]+"</td>";
                    var_value+=1*etv["converted_value"]*test_dic[test_number]["value"]/100;
                    var_weight+=1*test_dic[test_number]["value"]/100
                   } else{s+="<td style=';text-align:right;width:"+col_width+"px'></td>"}
                  }

                 try{
                     var ss_=""; var ss_var_value="";
                     if(var_value>0) {var_value=var_value/var_weight;
                         if(Math.round(100*var_value)/100 > up_value){ss_="background-color:"+up_color+";color:white;"}
                         else if (Math.round(100*var_value)/100 < down_value){ss_="background-color:"+down_color+";color:white;";
                         } else {ss_="background-color:"+other_color+";color:white;"}
                         ss_var_value = (Math.round(100*var_value)/100)
                     }
                 } catch(er){}
                 s+="<td style=';text-align:right;width:"+col_width+"px;"+ss_+"'>"+ss_var_value+"</td>";
                 s+="</tr>";
              }
         }

         entity_test_value["improve"]={}
         for(var h in entity_test_value["best"])
         {
          entity_test_value["improve"][h]={
           "value":1*entity_test_value["best"][h]["value"]-1*entity_test_value["worst"][h]["value"],
           "converted_value":1*entity_test_value["best"][h]["converted_value"]-1*entity_test_value["worst"][h]["converted_value"]
          }
         }

         var tt={"best":{},"worst":{},"improve":{}}
         for(var t_ in tt)
         {var ss="<tr><td style='width:"+userid_width+"px'>"+e_userid+"</td>"
          ss+="<td style='width:"+name_width+"px'>"+e_first_name+"</td>";
          ss+="<td style='width:"+name_width+"px'>"+e_last_name+"</td>";
          var var_value=0;var var_weight=0
          for(var test_number in test_dic)
          {var etv=entity_test_value[t_][""+test_number];
                   if(etv!=null && (1*etv["converted_value"])>-1)
                   {
                    ss+="<td v='"+etv["value"]+"' style=';text-align:right;width:"+col_width+"px'>"+etv["converted_value"]+"</td>";
                    var_value+=1*etv["converted_value"]*test_dic[test_number]["value"]/100;
                    var_weight+=1*test_dic[test_number]["value"]/100
                   } else{ss+="<td style=';text-align:right;width:"+col_width+"px'></td>"}
           }
           try{
             var ss_=""; var ss_var_value="";
             if(var_weight>0) {var_value=var_value/var_weight;
                 if(Math.round(100*var_value)/100 > up_value){ss_="background-color:"+up_color+";color:white;"}
                 else if (Math.round(100*var_value)/100 < down_value){ss_="background-color:"+down_color+";color:white;";
                 } else {ss_="background-color:"+other_color+";color:white;"}
                 ss_var_value = (Math.round(100*var_value)/100)
                 entity_test_value[t_]["final"]=ss_var_value;
             }
           } catch(er){}

           ss+="<td style=';text-align:right;width:"+col_width+"px;"+ss_+"'>"+ss_var_value+"</td>";
           ss+="</tr>";
           tt[t_]["s"]=ss;tt[t_]["tests"]=entity_test_value[t_]
         }
         //alert(e_userid+"\nBBB\n"+ s)
         tt["all"]={};tt["all"]["s"]=s
         return tt;
        }

      var tt_={"all":{"body":"", "final":{}}, "best":{"body":"", "final":{}},"worst":{"body":"", "final":{}},"improve":{"body":"", "final":{}}}
      var s_="<tr><th style='width:"+userid_width+"px'>UserId</th>"
      s_+="<th style='width:"+name_width+"px'>First Name</th>"
      s_+="<th style='width:"+name_width+"px'>Last Name</th>"
      //var s_all=s_; var s_best=s_;var s_worst=s_;var s_improve=s_;
      for(var k in tt_){eval('s_'+k+'=s_')}
      s_all+="<th style='width:"+event_date_width+"px'>Event Date</th>"
      //alert("90160-7\n"+JSON.stringify(tests_config));
      var st_=""
      for(var test_number in test_dic){
       var test_name=tests_config["test_name"][tests_config["test_number"].indexOf(""+test_number)];
       test_name=test_name.split("(")[0];
       st_+="<th style='width:"+col_width+"px'>"+test_name+"</th>";
      };
      st_+="<th style='width:"+col_width+"px'>"+var_name+"</th></tr>";

      for(var k in tt_){eval('tt_["'+k+'"]["thead"]=s_'+k+'+st_')}

      for(var i=0; i<n_group;i++)
      {
         try{var entity_number=parseInt(group_dic[i]);}catch(er){alert(er)}
         var e_idx=entity_config["id"].indexOf(""+entity_number);
         var e_first_name=entity_config["first_name"][e_idx];
         var e_last_name=entity_config["last_name"][e_idx];
         var e_userid=entity_config["userid"][e_idx];

         var dic_={"entity_number":entity_number, "entity_name":entity_name,
                   "result":result, "n_tests":n_tests, "e_first_name":e_first_name, "e_last_name":e_last_name,
                   "tests_config":tests_config,"test_dic":test_dic}
         var tt=ff(dic_, 0);
         for(var k in tt_){
          eval('tt_["'+k+'"]["body"]+=tt["'+k+'"]["s"]')

try{
      //alert(k+"\n"+JSON.stringify(tt[k]));
          eval('tt_["'+k+'"]["final"]['+entity_number+']=tt["'+k+'"]["tests"]["final"]')
} catch(er){}

     // alert(JSON.stringify(tt_[k]["final"]));

          //tt_[k]["final"][entity_number]=  tt[k]["tests"]["final"]   tt[k]["tests"]["final"]
         }
      }

      //alert(JSON.stringify(result));
      //alert(JSON.stringify(result)+"\n\n"+JSON.stringify(test_dic)+"\n\ntests_config:\n"+JSON.stringify(tests_config));
      //alert(this_obj.parent.tbody.outerHTML)
      this_obj.tests_reports=tt_
      this_obj.parent.tr_h.innerHTML=tt_["all"]["thead"];
      this_obj.parent.tbody.innerHTML=tt_["all"]["body"];
    }

    //-- need async to make this work --
    try{
    var pp_=this.parent.data["properties"]; var td = document.createElement("td");
    var img_=pp_["image"];if(img_==null || img_==""){img_="loading.png"}
    td.innerHTML="<img id='img_"+pp_["obj_number"]+"' src='/media/training/images/"+img_+"' style='width:300px;height:300x;'>";
    this.parent.tr_h.innerHTML=td.outerHTML;this.parent.tbody.innerHTML="";
    }catch(er){}
    //
    atm.activate_obj_function(fun, dic, this)
 }

}

// -- acAdmin --
function acAdmin(){}
acAdmin.prototype.create_obj = function(){this.creator.create_obj();}

//-- acToDo --
function acToDoCreator(parent){this.parent=parent;}
acToDoCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;var pp_=dic["properties"];
  var width_=pp_["width"]; if(width_==null || width_==""){width_=80};
  var height_=pp_["height"]; if(height_==null || height_==""){height_=300};
  var obj_number = dic["properties"]["obj_number"]
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
   //alert("90100-2 acGroupCreator.prototype.create_obj\n"+JSON.stringify(dic));
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"%"
  this.main_div.setAttribute("style", style_);
  this.main_div.setAttribute("container_id", dic["container_id"]);
  //--
  container.appendChild(this.main_div);
  this.add_task_btn=document.createElement("button");
  this.main_div.appendChild(this.add_task_btn);
  this.add_task_btn.innerHTML="Add Task";
  this.add_task_btn.my_creator_obj=this;
  this.add_task_btn.addEventListener("click", function(event){
    var e=event.target;var c=e.my_creator_obj;var t=c.table_;
    var dic=c.parent.data;
    var row_data={"record_id":"new",
                  "priority":100,
                  "subject":"",
                  "description":"",
                  "is_active":true
                  }
      c.get_new_row(row_data, dic, html_obj=t, n_place=1)
  })

  //--
  this.table_=document.createElement("table");
  this.table_.setAttribute("class", "basic")
  this.main_div.appendChild(this.table_);
  this.thead_=document.createElement("thead");
  this.table_.appendChild(this.thead_);
  this.thead_tr=document.createElement("tr");
  this.thead_.appendChild(this.thead_tr);
  var th=document.createElement("th");th.innerHTML="Priority";this.thead_tr.appendChild(th);
  var th=document.createElement("th");th.innerHTML="Subject";this.thead_tr.appendChild(th);
  var th=document.createElement("th");th.innerHTML="Description";this.thead_tr.appendChild(th);
  var th=document.createElement("th");th.innerHTML="Is Active";this.thead_tr.appendChild(th);
  var th=document.createElement("th");this.thead_tr.appendChild(th);
  //--
  this.tbody_=document.createElement("tbody");
  this.table_.appendChild(this.tbody_);
  //--
  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
 // alert(this.main_div.outerHTML)
  //--
  this.set_data();
}

acToDoCreator.prototype.get_new_row = function(row_data, dic, html_obj, n_place=null)
{
      var pp_=dic["properties"]; var content_type_="todo";
      //alert("90101-111-0 row_data\n"+JSON.stringify(pp_));
      if(n_place==null){var tr=document.createElement("tr");html_obj.appendChild(tr)}
      else{var tr=html_obj.insertRow(n_place)}
      tr.setAttribute("record_id", row_data["record_id"]);
      var td=document.createElement("td");tr.appendChild(td);
      var select=document.createElement("select"); td.appendChild(select);
      var o=document.createElement("option");select.appendChild(o);o.value=-1;o.innerHTML="----"
      var o=document.createElement("option");select.appendChild(o);o.value=1;o.innerHTML="High1";
      var o=document.createElement("option");select.appendChild(o);o.value=4;o.innerHTML="High2";
      var o=document.createElement("option");select.appendChild(o);o.value=6;o.innerHTML="High3";
      var o=document.createElement("option");select.appendChild(o);o.value=8;o.innerHTML="High4";
      var o=document.createElement("option");select.appendChild(o);o.value=10;o.innerHTML="High5";
      var o=document.createElement("option");select.appendChild(o);o.value=12;o.innerHTML="Mid1";
      var o=document.createElement("option");select.appendChild(o);o.value=14;o.innerHTML="Mid2";
      var o=document.createElement("option");select.appendChild(o);o.value=16;o.innerHTML="Mid3";
      var o=document.createElement("option");select.appendChild(o);o.value=20;o.innerHTML="Low1";
      var o=document.createElement("option");select.appendChild(o);o.value=22;o.innerHTML="Low2";
      var o=document.createElement("option");select.appendChild(o);o.value=24;o.innerHTML="Low3";
      select.setAttribute("style", "width:100px");
      var field="priority";select.setAttribute("field", field);
      select.setAttribute("container_id", dic["container_id"]);
      select.setAttribute("content_type", content_type_);
      select.value=row_data[field];

      var td=document.createElement("td");tr.appendChild(td);
      var ta=document.createElement("textarea");td.appendChild(ta);
      ta.setAttribute("rows", 3);ta.setAttribute("cols", 40);
      ta.setAttribute("container_id", dic["container_id"]);
      ta.setAttribute("content_type", content_type_);
      var field="subject";ta.setAttribute("field", field);ta.value=row_data[field];

      var td=document.createElement("td");tr.appendChild(td);
      var ta=document.createElement("textarea");td.appendChild(ta);
      ta.setAttribute("rows", 3);ta.setAttribute("cols", 40);
      var field="description";ta.setAttribute("field", field);
      ta.setAttribute("container_id", dic["container_id"]);
      ta.setAttribute("content_type", content_type_);
      ta.value=row_data[field];

      var td=document.createElement("td");tr.appendChild(td);
      var input=document.createElement("input");input.setAttribute("type", "checkbox");
      td.appendChild(input);
      var field="is_active";input.setAttribute("field", field);
      input.setAttribute("container_id", dic["container_id"]);
      input.setAttribute("content_type", content_type_);
      if(row_data[field]==true){input.checked=true}
      //input.value=row_data[field];

      var td=document.createElement("td");tr.appendChild(td);
      var b=document.createElement("button");td.appendChild(b);b.innerHTML="D";
      b.dic=dic;
      b.addEventListener("click", function(event){
          var id_ = prompt("Are you sure you want to delete this record? if so type y" , 'No')
          if(id_ != 'y') {return;};
          var e=event.target;var r=e.parentNode.parentNode
          record_id_=r.getAttribute("record_id");r.parentNode.removeChild(r);
          var container_id=dic["container_id"]
          var container = getEBI("content_"+container_id);
          var table_=container.getAttribute("table")

          var dic__={"model":table_, "record_id":record_id_}
          var fun=function(data, html_obj)
          {
           if(data["status"]=="ok")
           {alert("Record "+record_id_+" was deleted.")}else{alert("Error deleting record "+record_id_+".")};
          }
          atm.delete_record_from_table(fun, dic__, null)
       })
 }

acToDoCreator.prototype.set_data = function()
{
  var dic=this.parent.data;var pp_=dic["properties"];
  var container_id=dic["container_id"]
  var table_= getEBI("content_"+dic["container_id"]).getAttribute("table")
  var fun = function(data,ll){
    var obj=ll[0];var html_obj=obj.tbody_; var dic=ll[1]
     //alert("data=\n"+JSON.stringify(data));
     var n=data[data["pkf_name"]].length;
     for(var i=0;i<n;i++)
     {
      var row_data={"record_id":data["id"][i],
                    "priority":data["priority"][i],
                    "subject":data["subject"][i],
                    "description":data["description"][i],
                    "is_active":data["is_active"][i]
                    }
    //alert("90876-760-1\n"+JSON.stringify(row_data));

      obj.get_new_row(row_data, dic, html_obj)
     }
   }
   var dic_={"model": table_, "parent_model":"", "parent_id": "", "number_of_rows": "1000000",
               "fields": ["subject","description", "priority", "is_active"],
               "filters":{}, "order_by": {}}
  //alert("90876-764-4\n"+JSON.stringify(dic_));
  atm.get_data(fun, dic_, [this, dic]);
}

// -- acPlugin --
function acPlugin(){}
acPlugin.prototype.create_obj = function(){this.creator.create_obj();}

// -- acContainerCreator --
function acContainerCreator(parent){this.parent=parent;this.tab=parent.tab}
acContainerCreator.prototype.create_obj = function()
{
  this.link_dic=this.parent.data;
  //alert("9065 this.tab.tab_name: "+this.tab.tab_name)
  var pp_=this.link_dic["properties"]
  //alert("this.link_dic="+JSON.stringify(pp_))
  //--
  var container = document.getElementById("content_"+this.link_dic["container_id"]);
  //alert(container.outerHTML)
  //--
  this.obj_number=pp_["obj_number"]
  this.link_number=pp_["obj_number"]
  this.link_content=document.createElement("div");
  this.link_content.my_creator_obj=this
  this.link_content.setAttribute("id", this.obj_number);
  this.link_content.setAttribute("link_number", this.obj_number);
  this.link_content.setAttribute("container_id", this.link_dic["container_id"]);
  this.link_content.setAttribute("obj_type", this.link_dic["obj_type"]);

  this.link_content.tab=this.parent.tab;
  var width_=100;if((pp_["width"]!=null && pp_["width"]!="")){width_=pp_["width"]}
  var height_=100;if((pp_["height"]!=null && pp_["height"]!="")){height_=pp_["height"]}

  var background_color_="#AEEEEF";if(pp_["background-color"]!=null && pp_["background-color"]!=""){background_color_=pp_["background-color"]}
  var color_="black";if((pp_["color"]!=null && pp_["color"]!="")){color_=pp_["color"]}
  var ss_="position: relative;left:"+pp_["x"]+"px;top:"+pp_["y"]+"px;color:"+color_
  //var ss_="position: relative;left:"+pp_["x"]+"px;top:"+pp_["y"]+"px;background-color:transparent;color:"+color_
  ss_+=";width: "+width_+"px;height:"+height_+"px;display:block;"
  if(pp_["border_style"]!=null && pp_["border_style"]!=""){ss_+="border-style:"+pp_["border_style"]+";"}
  if(pp_["border_width"]!=null && pp_["border_width"]!=""){ss_+="border-width:"+pp_["border_width"]+"px;"}
  if(pp_["border_color"]!=null && pp_["border_color"]!=""){ss_+="border-color:"+pp_["border_color"]+";"}
  if(pp_["border_radius"]!=null && pp_["border_radius"]!=""){ss_+="border-radius:"+pp_["border_radius"]+"px;"}

  if(pp_["z_index"]!=null && pp_["z_index"]!=""){
    if(1*pp_["z_index"]<0){
     ss_+="background-color:transparent;"
    } else {ss_+="z-index:"+pp_["z_index"]+";background-color:"+background_color_+";"}

  }

  this.link_content.setAttribute("style", ss_);
  this.link_content.my_creator_obj=this
  container.appendChild(this.link_content);
  this.link_content.onclick=container_on_click;
  this.link_content.container_on_change=container_on_change;
  //alert("9066 this.tab.tab_name: "+this.tab.tab_name)
  this.process_content();
}

// 220000
var process_content=function(){
//alert("9087")
  //alert(JSON.stringify(this.link_dic["functions"]));
  for(f in this.link_dic["functions"])
  {
    if(f!="onclick"){var s='this.link_content.'+f+'='+this.link_dic["functions"][f];eval(s);}
  }
  for (i in this.tab.tab_objects[this.link_number])
  {
   dic_=this.tab.tab_objects[this.link_number][i];
   //alert(JSON.stringify(dic_));
   this.tab.generate_obj(dic=dic_);
  }
}
var container_on_change=function (event){
  event.stopPropagation();
  var e=event.target;
  //alert("c9001-1 "+e.outerHTML);
  var e_container_id_=e.getAttribute("container_id")
  var container = getEBI("content_"+e_container_id_)
  //alert("c9002 container \n"+container.outerHTML) //alert("9050 " + JSON.stringify(container.my_creator_obj.link_dic))
  var c_container_id_=this.getAttribute("link_number")
  //  alert("c9002-1  e_container_id_=" + e_container_id_+ " c_container_id_=" + c_container_id_)
  //alert("c9002-2 container: e.outerHTML: "+e.outerHTML+"\n\n e.value= " + e.value+"\n\n this.outerHTML= " + this.outerHTML)
  //if(e_container_id_!=c_container_id_) {return;}
  var foreign_table=e.getAttribute("foreign_table")
  var dependent=e.getAttribute("dependent")
  var field_=e.getAttribute("field");
  if(field_=="" || field_==null){return}
  try{var type_=e.getAttribute("type");if(type_==null || type_==""){type_=""}} catch(er){}
  var field__="";var v__="";
  var need_to_return = false;
  //alert(dependent)
  if(dependent!=null && dependent!="")
  {
   var ll_dependents =dependent.split(",")
   if(container.dependents==null){container.dependents={}}
   var vv=e.value;if(type_=="date"){var ss=vv.split("-");vv=ss[0]+ss[1]+ss[2]}
   if(!(field_ in container.dependents))
   {
     container.dependents[field_]={"value":vv, "dependent":field_};
     //alert('90806-68 = \n'+ JSON.stringify(container.dependents)+"\n\n"+JSON.stringify(dependent));
     for(var j in ll_dependents){var k=ll_dependents[j];if(!(k in container.dependents)){need_to_return=true;}}
   }
   else{field__=field_; v__=vv}
  }
  // try{alert('c9002-5 = \n'+ JSON.stringify(container.dependents)+"\n\n"+JSON.stringify(dependent));}catch(er){}
  if(foreign_table!=null && foreign_table!="")
  {
   if(container.foreign_keys==null){container.foreign_keys={}}
   var vv=e.value;if(type_=="date"){var ss=vv.split("-");vv=ss[0]+ss[1]+ss[2]}
   if(!(field_ in container.foreign_keys))
   {container.foreign_keys[field_]={"value":vv, "foreign_table":foreign_table};need_to_return=true;}
   else{field__=field_; v__=vv}
  }
  //alert(need_to_return)
  if(need_to_return==true){return}
  //alert("c9002-7: " + field_)
  container.tab.parent.active_tab_content=container;
  try{var content_type_=e.getAttribute("content_type");if(content_type_==null || content_type_==""){content_type_=""}} catch(er){}
  var html_obj_to_update=container
  //alert("c9002-8  e: " + e.outerHTML)
  //alert("c9002-9  "+content_type_)
  var field__u=field_; var v__u=e.value;
  var type__=e.getAttribute("type_");
  //alert("c9002-10 type__="+type__)
  if(type__!=null && type__!=""){
      if(type__=="Select"){
       var s__="";
       for(j in e.selectedOptions)
       {
        try{if(e.selectedOptions[j]!=null && e.selectedOptions[j].tagName.toLowerCase()=="option")
        {if(s__!=""){s__+=","};s__+=e.selectedOptions[j].getAttribute("value")}} catch(er){}
       }
       v__u=s__
       //alert("9052 v__u = "+v__u)
      } else if (type__=="RichText"){
       v__u=e.innerHTML
      //alert(v__u)
      }
  }
  if(field__!=""){field__u=field__; v__u=v__}

  //alert("c9002-15  field__u= "+field__u+" - v__u= "+v__u)
  if (content_type_=="accounting")
  {
    try{
    var record_id_=e.getAttribute("record_id");
    var parent_id_=container.getAttribute("record_id");
    // alert("9084 record_id_= "+record_id_+" : parent_id_= "+parent_id_)
    if(record_id_==null || record_id_==""){record_id_="new"}} catch(er){}
    var model_=e.getAttribute("model")
    var parent_model_=container.my_creator_obj.link_dic["properties"]["table"];
    var html_obj_to_update=e.parentNode.parentNode;
    var account=e.getAttribute("account")
    var dic_data={"model":model_, "parent_model":parent_model_, "pkey":record_id_, "parent_pkey":parent_id_,
                  "fields": {"account":account}, "type":type_, "foreign_keys":{}, "dependents":{}}
        dic_data["fields"][field__u]=v__u;

    //alert('9080 dic_data = '+ JSON.stringify(dic_data));
    container.tab.parent.save_data(html_obj_to_update, dic_data);
  } else if (content_type_=="activity"){
   //alert(e.outerHTML)
   var model_=e.getAttribute("model")
   html_obj_to_update=e.parentNode.parentNode;
  //alert(html_obj_to_update.outerHTML)
   var parent_id_ = e.getAttribute("parent_id")
   var record_id_ = html_obj_to_update.getAttribute("record_id")
   var parent_model_=e.getAttribute("parent_model")
   var fields_values=e.getAttribute("fields_values");var fvs = fields_values.split("-")
   //alert('9080-555 dic_data= '+ JSON.stringify(fvs));
   var dic_data={"model":model_, "parent_model":parent_model_, "pkey":record_id_, "parent_pkey":parent_id_,
                    "fields": {}, "type":type_, "foreign_keys":container.foreign_keys, "dependents":container.dependents}
   for(var q in fvs){var x=fvs[q].split("__");dic_data["fields"][x[0]]=x[1];}
   //alert('9080-222 dic_data= '+ JSON.stringify(dic_data));
   //alert(html_obj_to_update.outerHTML)
   container.tab.parent.save_data(html_obj_to_update, dic_data);
  }
   else if (content_type_=="todo")
  {
    try{
    var html_obj_to_update=e.parentNode.parentNode
    var record_id_=html_obj_to_update.getAttribute("record_id");
    var parent_id_="";var parent_model_="";
    if(record_id_==null || record_id_==""){record_id_="new"}} catch(er){}
    var model_=container.getAttribute("table")
    //var html_obj_to_update=e
    var dic_data={"model":model_, "parent_model":parent_model_, "pkey":record_id_, "parent_pkey":parent_id_,
                  "fields": {}, "type":type_, "foreign_keys":{}, "dependents":{}}
                  if(type_=="checkbox"){if(e.checked==true){v__u="True"}else{v__u="False"}}
        dic_data["fields"][field__u]=v__u;;
    //alert('9080 dic_data = '+ JSON.stringify(dic_data));
    container.tab.parent.save_data(html_obj_to_update, dic_data);
  }
   else if (content_type_=="data_entry")
  {
    //alert("c9002-50")
  } else
  {
     var model_=container.my_creator_obj.link_dic["properties"]["table"];
     var record_id_=container.getAttribute("record_id");
     var parent_id_=container.getAttribute("parent_id");
     var is_plugin = false;
     if(record_id_=="plugin"){is_plugin = true;
         //alert(e.outerHTML);
         if(type_=="radio"){html_obj_to_update=e.parentNode.parentNode}else {html_obj_to_update=e};
         //alert("html_obj_to_update:\n"+html_obj_to_update.outerHTML);
         record_id_=html_obj_to_update.getAttribute("record_id");
         var fields_values=e.getAttribute("fields_values");var fvs = fields_values.split("-")
       }

     //alert("9055-35:\nmodel_="+model_ +"\nrecord_id_="+record_id_)
     if(this.my_creator_obj.is_json_data){
         // alert(model_+"\nparent_id:"+parent_id_+"\nrecord_id:"+record_id_+"\nfield:"+field__u+"\nfield value:"+v__u+"\n"+event.target.outerHTML)
         var json_manager_obj_number = container.getAttribute("json_manager_obj_number");
         // alert(json_manager_obj_number)
         var jmc = get_creator(json_manager_obj_number)
         var dic_data={"model": model_,"parent_id":parent_id_,"record_id":record_id_,"field":field__u,"field_value":v__u,
                      "container_id":e_container_id_}
         // alert('9083 dic_data= '+ JSON.stringify(dic_data));
         jmc.update_json_record(dic_data);
       } else {
        try{var parent_model_=container.my_creator_obj.link_dic["properties"]["parent_table"]} catch(er){};
        //alert("c9002-100 parent_model_=  "+parent_model_+" model_= "+model_+" parent_id_="+parent_id_+" record_id_="+record_id_+" fields_values="+fields_values)
        if(parent_model_==null){var parent_model_="";}
        //alert("c9002-105 else: record_id_ "+record_id_ +" parent_id_= "+parent_id_+" model_= "+model_+" parent_model_= "+parent_model_)
        var foreign_keys_ = container.foreign_keys; var dependents_=container.dependents;
        //alert(e.outerHTML)
        fields_={}
        if(type__=="tests" || type__=="groups" || type__=="variables" || type__=="recording")
        {
            foreign_keys_={};parent_model_=model_;parent_id_=record_id_;html_obj_to_update=e;
            var model_=e.getAttribute("table");var atm_=this.tab.parent;
            if(type__=="recording"){
                //alert("c9002-109-1\n\n"+e.outerHTML)
                if(e.getAttribute("type")=="checkbox"){v__u = ""+e.checked;}
                var record_id_=e.getAttribute("record_id"); var id_ll=e.getAttribute("id").split("_");
                foreign_keys_[e.getAttribute("vertical_field")]={"value":id_ll[1], "foreign_table":e.getAttribute("vertical_field")+"s"},
                foreign_keys_[e.getAttribute("horizontal_field")]={"value":id_ll[2], "foreign_table":e.getAttribute("horizontal_field")+"s"}
             } else {
               dependents_={};
               var record_id_=e.getAttribute("record_id");
               fields_[e.getAttribute("value_field")]=e.getAttribute(e.getAttribute("value_field"))
               if(type__=="tests" || type__=="groups"){v__u = ""+e.checked;}
               //alert('9080-111-11 fields_= '+ JSON.stringify(field_));
               //alert("c9001-55-0 \nparent_model_="+parent_model_+"\nparent_id_="+parent_id_+"\nmodel_="+model_+"\nrecord_id_="+record_id_)
             }
             if(((type__=="tests" || type__=="groups") && (e.checked==false)) ||
                ((type__=="variables" || type__=="recording") && (e.value=="" || Math.round(parseInt(e.value))==0 || v__u=="false"))){
                 if(model_!=null && model_!="" && String(model_)!="undefined" ){
                     var dic_={"obj":"AdvancedTabs","atm":atm_.my_name,"app":atm_.my_app,"fun":"delete_record",
                               "params": {"app":atm_.my_app,"model":model_,"id":record_id_}}

                     //alert('9080-1222-11 fields_= '+ JSON.stringify(dic_));
                     //alert(JSON.stringify(dic_));
                     $.post(atm_.activate_function_link_,
                      {dic : JSON.stringify(dic_)},
                      function(dic){
                        id_ = dic["result"]["id"];
                        //alert("record was deleted: "+id_)
                     })
                   }
                 e.setAttribute("record_id", "new");
                 try{var n=e.getAttribute("my_creator_number");var c=get_creator(n);c.on_record_created_deleted(e,action="deleted")} catch(er){}
                 return;
             }
         }
         //alert("c9002-110")
         //alert("c9002-110-1 \nmodel_="+model_+"\nparent_model_="+parent_model_+"\nrecord_id_="+record_id_+"\nparent_id_="+parent_id_+"")
           //alert("c9002-110-2 fields_=\n"+JSON.stringify(fields_))
           //alert("c9002-110-3 type_=\n"+type_)
           //alert("c9002-110-4 foreign_keys_=\n"+JSON.stringify(foreign_keys_))
           //alert("c9002-110-5 dependents_=\n"+JSON.stringify(dependents_))
           //alert("c9002-110-6 id=\n"+e.getAttribute("id"))
           var dic_data={"model":model_, "parent_model":parent_model_, "pkey":record_id_, "parent_pkey":parent_id_,
                        "fields": fields_, "type":type_, "foreign_keys":foreign_keys_, "dependents":dependents_,
                        "element_id":e.getAttribute("id")}
           //alert("c9002-120 dic_data= "+ JSON.stringify(dic_data));
           if(is_plugin==true){for(var q in fvs){var x=fvs[q].split("__");dic_data["fields"][x[0]]=x[1];}}
           else {dic_data["fields"][field__u]=v__u;}
           //alert('9081-222 dic_data= '+ JSON.stringify(dic_data));
           //alert(html_obj_to_update.outerHTML)
           dic_data["for_refresh_my_tables"]={"f":field__u, "v":v__u, "container_id":e_container_id_}
           container.tab.parent.save_data(html_obj_to_update, dic_data);

           //alert(c9002-130+e.value)
       }
  }
  // event.stopPropagation();
}
var container_on_click=function(event){
      //alert("90123-10:\n"+event.target.outerHTML)
      event.stopPropagation();
      this.tab.parent.active_tab_content=this;
      var e=event.target;
      if(this.tab.parent.new_obj_to_create){
            this.tab.new_obj_to_create=this.tab.parent.new_obj_to_create;
            this.tab.parent.new_obj_to_create=null;
        }

      if(this.tab.new_obj_to_create==null){
         var click_event=new Event("click", {bubbles: true});
         if(event.ctrlKey){
           //alert("90123-1:\n"+event.target.outerHTML)
           var obj_type=e.getAttribute("obj_type");
           try{while(obj_type==null){e=e.parentNode;var obj_type=e.getAttribute("obj_type");}} catch(er){}
            if(obj_type==null){
            //alert(this.tab.parent.editor.main_menus["TabContent"].btn.outerHTML)
              this.tab.parent.editor.main_menus["TabContent"].btn.dispatchEvent(click_event);
              //alert(66666666661)
              //alert(this.tab.parent.editor.main_menus["TabContent"].my_sub_objs["content"].btn.outerHTML)
              this.tab.parent.editor.main_menus["TabContent"].my_sub_objs["content"].btn.dispatchEvent(click_event)
              //alert(777777777771)
            }
            else{
             var obj_number=e.getAttribute("id")
             var container_id=e.getAttribute("container_id")
             var click_event=new Event("click", {bubbles: true});
             var dic__=this.tab.tab_objects[container_id][obj_number];
             this.tab.parent.editor.main_menus[dic__["parent_obj_name"]].btn.dispatchEvent(click_event);
             var s='this.tab.active_obj=this.tab.parent.get_obj(this.tab,dic__);'
             //alert(s)
             eval(s)
             this.tab.active_obj.create_editor();
             }
         } else {
               var s=f+this.link_number+'='+link_dic["functions"]["onclick"]
               //alert(s);
               try{eval(s);eval(f+this.link_number+'(event)');} catch(er){}
         }
      } else {

         //alert("CCCCCC e="+e.outerHTML+" \nthis="+this.outerHTML)
         //alert(JSON.stringify(this.tab.new_obj_to_create));

        var dic=this.tab.new_obj_to_create;
        var x=event.clientX-e.offsetLeft;
        //alert(event.clientX); //alert(e.offsetLeft); //alert(x)
        //alert(event.clientX); //alert(event.pageX); //-e.offsetLeft;
        if(e.parentNode.parentNode.offsetTop*1==0){var y=event.clientY-e.offsetTop;}
        else{var y=event.clientY-e.parentNode.parentNode.offsetTop;}
        // alert('event.clientY'); // alert(event.clientY);// alert('e.offsetTop');
        //alert(e.offsetTop);//alert('e.parentNode.parentNode.offsetTop')
        //alert(e.parentNode.parentNode.offsetTop); //alert('y');//alert(y)

        var obj_number=this.tab.get_next_obj_number();
        var container_id=e.getAttribute("link_number");
        dic["properties"]["obj_number"]=obj_number;dic["container_id"]=container_id;
        dic["properties"]["x"]=x;dic["properties"]["y"]=y;

        if(dic["obj_type"]=="PopWin")
        {
          var popup_name_=prompt("Enter name for new Popup win:",'');if(popup_name_==''){alert("Please enter a name for popup win"); return;}
          var dic_={"properties":{"id":obj_number, "link_number":obj_number, "container_id":container_id, "tab_id":this.tab.tab_id,"name":popup_name_,"title":popup_name_,"zindex":50,"height":"500","width":"500","right":"25%","top":"25%",
                    "background_color":"white", "title_color": "#fff", "title_background_color": "#2196F3", "is_panel":"true"},
                     "functions":{"__init___":"function(win_obj){\ntry{\n\n} catch(er){alert('er903: '+ er)}}",
                                  "__get_panel_structure___":"function(win_obj){\ntry{\nvar dic={};\n\nwin_obj.buttons=dic;\n} catch(er){alert('er904: '+ er)}}",
                                  "__set_panel___":"function(win_obj){\ntry{\n\n} catch(er){alert('er906: '+ er)}}"
                                  },
                     "popwin_content":{"properties":{"link_number":obj_number, "container_id":container_id, "content_type": "simple", "width":"400","table":""},
                                       "functions":{}}
                   };

             //alert('9025: dic_\n:'+JSON.stringify(dic_))
             var win_obj = this.tab.get_pop_win_obj(dic_)
             try{
               win_obj.__init__();
               win_obj.set_win_frame_style(dic_["properties"]["zindex"], dic_["properties"]["height"], dic_["properties"]["width"], dic_["properties"]["right"], dic_["properties"]["top"], dic_["properties"]["background_color"])
               win_obj.set_acWinStatEventListeners(this.tab.parent.editor);
             } catch(er){alert('er90351: '+ er)}
        } else{
          this.tab.active_obj=this.tab.generate_obj(dic=dic);
          this.tab.active_obj.create_editor();
          if(this.tab.tab_objects[container_id]==null){this.tab.tab_objects[container_id]={}}
          this.tab.tab_objects[container_id][obj_number]=this.tab.active_obj.data;
        }
        this.tab.parent.save();
      }
}

acContainerCreator.prototype.process_content = process_content;

// UploadFile --
function acUploadFileCreator(parent){this.parent=parent;}

acUploadFileCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //alert("90352 "+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  var title = dic["properties"]["title"]
//  var field = dic["properties"]["field"]
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  this.div_=document.createElement("div");
  this.div_.my_creator_obj=this
  this.div_.setAttribute("obj_type", dic["obj_type"]);
  this.div_.setAttribute("id", obj_number);
//  this.div_.setAttribute("field", field);
  this.div_.setAttribute("container_id", dic["container_id"]);
  this.div_.setAttribute("type", dic["element_name"]);
  if("sheet_name" in dic["properties"]){var sheet_name_=dic["properties"]["sheet_name"]} else {var width_="Data"}
  if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
  if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
  var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  style_+="width:"+width_+"px;height:"+height_;
  this.div_.setAttribute("style", style_);
  container.appendChild(this.div_);
  // --
  this.input_file_=document.createElement("input");
  this.input_file_.setAttribute("id", "input_file_"+obj_number);
  this.input_file_.setAttribute("type", "file");
  this.input_file_.setAttribute("class", "filestyle");
  this.input_file_.setAttribute("data-icon", "true");
  this.input_file_.setAttribute("data-buttonText", "Brows");
  this.input_file_.setAttribute("name", "driver_file");
  this.div_.appendChild(this.input_file_);
  // --
  var span1=document.createElement("span");
  span1.setAttribute("id", "driver-uploader-success-alert");
  span1.setAttribute("class", "text-success");
  span1.setAttribute("style", "display:none");
  span1.innerHTML="Success!"
  this.div_.appendChild(span1);
  var span2=document.createElement("span");
  span2.setAttribute("id", "driver-uploader-failure-alert");
  span2.setAttribute("class", "text-danger");
  span2.setAttribute("style", "display:none");
  span2.innerHTML="text-danger"
  this.div_.appendChild(span2);
  this.input_file_.atm=this.parent.atm
  this.input_file_.dic=dic
  this.input_file_.onchange = function(event){
            var e = event.target;
            var fn = "";try{var fn = e.getAttribute("file_name");} catch(er){}
            var file = this.files[0];
            var filename = ''; if ('name' in file) {filename= file.name;} else {filename = file.fileName;}
            //var ext = filename.split(".");ext=ext[ext.length-1]
            if(fn !="" && fn != null){var ll=filename.split(".");filename=fn+"."+ll[ll.length-1]}
            e.setAttribute("filename", filename)
            var xhr = new XMLHttpRequest();
            try{xhr.dic=this.dic} catch(er){alert(er)}
            (xhr.upload || xhr).addEventListener('progress', function(e) {
                var done = e.position || e.loaded
                var total = e.totalSize || e.total;
                console.log('xhr progress: ' + Math.round(done/total*100) + '%');
            });
            xhr.addEventListener('load', function(e) {
                data = JSON.parse(this.responseText);
                    //alert("90353 "+JSON.stringify(data))
                    //alert(this.responseText)
                    try{var s="zz="+this.dic["functions"]["on_loaded"];eval(s);zz(data)} catch(er){}
                if(this.status != 200){
                  $('#driver-uploader-failure-alert').show();
                  return;
                }
                $('#driver_source').val('Flat file on idisas');
                $('#driver_name').val(data['file_remote_path']).blur();
                $('#driver-uploader-success-alert').show();
            });

            //alert(this.atm.upload_file_link_)
            xhr.open('post', this.atm.upload_file_link_, true);
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
            var fd = new FormData();
            fd.append("filename", filename);
            fd.append("drive_file", file);
            fd.append("app", this.atm.my_app);
            fd.append("obj_name", this.dic["properties"]["obj_name"]);
            fd.append("function_name", this.dic["properties"]["function_name"]);
            fd.append("topic_id", this.dic["properties"]["topic_id"]);
            fd.append("folder_type", this.dic["properties"]["folder_type"]);
            fd.append("entity_name", this.dic["properties"]["entity_name"]);

            fd.append("dimensions", this.dic["properties"]["dimensions"]);
            fd.append("fields", this.dic["properties"]["fields"]);
            fd.append("fact_model_field", this.dic["properties"]["fact_model_field"]);
            fd.append("sheet_name", this.dic["properties"]["sheet_name"]);
            fd.append("user_id", atm.user_id);
            //--
            var container = document.getElementById("content_"+this.dic["container_id"]);
            var record_id = container.getAttribute("record_id")
            var table = container.getAttribute("table")
            fd.append("record_id", record_id);
            // alert("903572-1 "+JSON.stringify(this.dic["properties"]));
            var field = this.dic["properties"]["field"]
            fd.append("model_field_name", field);
            fd.append("model_name", table);
            //--
            xhr.send(fd);
            try{var s="zz="+dic["functions"]["onchange"];eval(s);zz(event)} catch(er){}
  }

  // alert("90321-2 "+JSON.stringify(dic["functions"]));

  for(var f in dic["functions"])
  {if((f!="onchange") && (f!="on_loaded")){var s="this.input_file_."+f+"="+dic["functions"][f];eval(s);}}
}

acUploadFileCreator.prototype.create_obj_bu = function()
{
  function sendData(this_obj) {

    // If there is a selected file, wait it is read
    // If there is not, delay the execution of the function

    if (!this_obj.file.binary && this_obj.file.dom.files.length > 0) {

    setTimeout(sendData, 10, this_obj);return;}

    // To construct our multipart form data request,
    // We need an XMLHttpRequest instance
    const XHR = new XMLHttpRequest();
    //alert(csrftoken)

    // We need a separator to define each part of the request
    const boundary = "blob";
    // Store our body request in a string.
    let data = "";
    // So, if the user has selected a file

    if (this_obj.file.dom.files[0]) {
      // Start a new part in our body's request
      // Describe it as form data
      // Define the name of the form data
      // Provide the real name of the file
      data += '--${boundary}\r\ncontent-disposition: form-data; name="${this_obj.file.dom.name}"; filename="${this_obj.file.dom.files[0].name}"\r\n;'
      // And the MIME type of the file
      // There's a blank line between the metadata and the data
      data += 'Content-Type: ${this_obj.file.dom.files[0].type}\r\n\r\n';
      // Append the binary data to our body's request
      data += this_obj.file.binary + '\r\n';
    }

      // Text data is simpler
      // Start a new part in our body's request
      // Say it's form data, and name it
      // There's a blank line between the metadata and the data
      data += '--${boundary}\r\ncontent-disposition: form-data; name="${this_obj.text_.name}"\r\n\r\n';
      // Append the text data to our body's request
      data += this_obj.text_.value + "\r\n";
      // Once we are done, "close" the body's request
      data += '--${boundary}--';
      // Define what happens on successful data submission
    // Define what happens on successful data submission
    XHR.addEventListener('load', (event) => { alert('Yeah! Data sent and response loaded.');});
    // Define what happens in case of error
    XHR.addEventListener('error', (event) => { alert('Oops! Something went wrong.');});
    // Set up our request
    XHR.open('POST', this_obj.parent.atm.upload_file_link_, true); // 'https://example.com/cors.php');
    // Add the required HTTP header to handle a multipart form data POST request
    XHR.setRequestHeader('Content-Type', `multipart/form-data; boundary=${boundary}`);
    XHR.setRequestHeader("X-CSRFToken", csrftoken);
    XHR.onload = function (evt)
    {
     alert("returned from server" + evt)
    }
    XHR.send(data);// Send the data;
  }

  var dic=this.parent.data;
  //alert("90354 "+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  var title = dic["properties"]["title"]
  var container = document.getElementById("content_"+dic["container_id"]);
  //--

  this.form_=document.createElement("form");
  this.form_.setAttribute("obj_type", dic["obj_type"]);
  this.form_.setAttribute("id", obj_number);
  this.form_.setAttribute("container_id", dic["container_id"]);
  this.form_.setAttribute("type", dic["element_name"]);
  if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
  if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
  var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  style_+="width:"+width_+"px;height:"+height_;
  this.form_.setAttribute("style", style_);

  //alert(this.form_.outerHTML)

//  var to_1 = document.getElementsByName("csrfmiddlewaretoken")[0]
//  this.to_=document.createElement("input");
//  this.to_.setAttribute("name", "csrfmiddlewaretoken");
//  this.to_.setAttribute("value", to_1.value);
//  this.to_.setAttribute("type","hidden");
//  this.form_.appendChild(this.to_);


  var label_=document.createElement("label");label_.setAttribute("for","text_"+obj_number);this.form_.appendChild(label_);
  this.text_=document.createElement("input");this.text_.setAttribute("id","text_"+obj_number);
  this.text_.setAttribute("name", "text_"+obj_number);
  this.text_.setAttribute("type","text");
  this.form_.appendChild(this.text_);

  var br=document.createElement("br");
  this.form_.appendChild(br);
  var label_=document.createElement("label");label_.setAttribute("for","file_"+obj_number);this.form_.appendChild(label_);
  this.file_=document.createElement("input");this.file_.setAttribute("id","file_"+obj_number);
  this.file_.setAttribute("name", "name_"+obj_number);
  this.file_.setAttribute("type","file");
  this.form_.appendChild(this.file_);

  var submit_btn=document.createElement("button");submit_btn.innerHTML=title;
  this.form_.appendChild(submit_btn);
  container.appendChild(this.form_);

  this.file = {dom: this.file_, binary: null,}

  this.reader= new FileReader();
  this.reader.file=this.file;
  this.reader.onload = function(event) {var e=event.target;this.file.binary=e.result;};

  this.file_.reader=this.reader;
  this.file_.onchange = function(event) {
    if (this.reader.readyState === FileReader.LOADING) {this.reader.abort();}
    this.reader.readAsBinaryString(event.target.files[0]);
  };
  //alert(this.form_.outerHTML)
  this.form_.addEventListener('submit', (event) => {event.preventDefault();
  sendData(this);
  });


}


// -- Image --
function acImageCreator(parent){this.parent=parent;}

acImageCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //alert("903542 "+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  var title = dic["properties"]["title"]
//  var field = dic["properties"]["field"]
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  var pp_=dic["properties"];
  this.img_=document.createElement("img");
  this.img_.my_creator_obj=this;this.img_.container=container;this.img_.dic=dic;
  this.img_.setAttribute("obj_type", dic["obj_type"]);
  this.img_.setAttribute("id", obj_number);
//  this.div_.setAttribute("field", field);
  this.img_.setAttribute("container_id", dic["container_id"]);
  this.img_.setAttribute("type", dic["element_name"]);
  this.img_.setAttribute("alt", dic["properties"]["title"]);
  if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
  if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
  var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  style_+="width:"+width_+"px;height:"+height_+"px;";
  if(pp_["border_radius"]!=null && pp_["border_radius"]!=""){style_+="border-radius:"+pp_["border_radius"]+"px;"}
  if(pp_["border_width"]!=null && pp_["border_width"]!="" && pp_["border_color"]!=null && pp_["border_color"]!="")
  {style_+="border: "+pp_["border_width"]+"px solid "+pp_["border_color"]}
  this.img_.setAttribute("style", style_);
  this.img_.setAttribute("src", "/media/"+this.parent.atm.my_app+"/images/"+dic["properties"]["src"]);
  container.appendChild(this.img_);
  this.img_.onclick = function(event){
      if(event.ctrlKey && event.altKey){
        var div_=document.createElement("div");
        var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
        style_+="width:400px;height:400px";
        div_.setAttribute("style", style_);
        this.container.appendChild(div_);
        var input_file_=document.createElement("input");
        input_file_.setAttribute("id", "input_file_"+obj_number);
        input_file_.setAttribute("type", "file");
        input_file_.setAttribute("class", "filestyle");
        input_file_.setAttribute("data-icon", "true");
        input_file_.setAttribute("data-buttonText", "Brows");
        input_file_.setAttribute("file_name", dic["properties"]["src"]);
        input_file_.setAttribute("name", "driver_file");
        div_.appendChild(input_file_);
        input_file_.atm=this.my_creator_obj.parent.atm
        input_file_.dic=this.dic;
        input_file_.div_=div_;

        input_file_.onchange = function(event){
                var e = event.target;
                var filename = "";try{filename = e.getAttribute("file_name");} catch(er){}
                var file = this.files[0];
                var xhr = new XMLHttpRequest();
                try{xhr.dic=this.dic} catch(er){alert(er)}
                (xhr.upload || xhr).addEventListener('progress', function(e) {
                    var done = e.position || e.loaded
                    var total = e.totalSize || e.total;
                    console.log('xhr progress: ' + Math.round(done/total*100) + '%');
                });
                xhr.div_=this.div_;
                xhr.addEventListener('load', function(e) {
                    data = JSON.parse(this.responseText);
                    location.reload();
                    this.div_.outerHTML="";
                    try{var s="zz="+this.dic["functions"]["on_loaded"];eval(s);zz(data)} catch(er){}
                });

                //alert(this.atm.upload_file_link_)
                xhr.open('post', this.atm.upload_file_link_, true);
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                var fd = new FormData();
                fd.append("filename", filename);
                fd.append("drive_file", file);
                fd.append("app", this.atm.my_app);
                fd.append("obj_name", this.dic["properties"]["obj_name"]);
                fd.append("function_name", this.dic["properties"]["function_name"]);
                fd.append("topic_id", "images");
                fd.append("folder_type", "media");
                fd.append("dimensions", "");
                fd.append("fields", "");
                fd.append("fact_model_field", "");
                //--
                fd.append("model_name", "");
                //--
                fd.append("user_id", "");
                fd.append("sheet_name", "");
                fd.append("entity_name", "");
                //--
                xhr.send(fd);
                try{var s="zz="+dic["functions"]["onchange"];eval(s);zz(event)} catch(er){}
      }
      }
  }
    // alert("90321-2 "+JSON.stringify(dic["functions"]));
  for(var f in dic["functions"])
  {if((f!="onchange") && (f!="on_loaded")){var s="this.input_file_."+f+"="+dic["functions"][f];eval(s);}}
}


// -- Video --
function acVideoCreator(parent){this.parent=parent;}

acVideoCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
//  alert("903542 "+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  var title = dic["properties"]["title"]
//  var field = dic["properties"]["field"]
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  this.vid_=document.createElement("video");
  this.vid_.controls = "control";
  this.src_=document.createElement("source");
//  "https://archive.org/download/C.E.PriceCatWalksTowardCamera/cat_walks_toward_camera_512kb.mp4"
  this.src_.setAttribute("src",  "/media/"+this.parent.atm.my_app+"/video/"+dic["properties"]["src"]);
  this.src_.setAttribute("type", "video/mp4");
  this.vid_.appendChild(this.src_);
  this.vid_.my_creator_obj=this;this.vid_.container=container;this.vid_.dic=dic;
  this.vid_.setAttribute("obj_type", dic["obj_type"]);
  this.vid_.setAttribute("id", obj_number);
//  this.div_.setAttribute("field", field);
  this.vid_.setAttribute("container_id", dic["container_id"]);
  this.vid_.setAttribute("type", dic["element_name"]);
  this.vid_.setAttribute("alt", dic["properties"]["title"]);
  if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
  if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
  var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  style_+="width:"+width_+"px;height:"+height_+"px;";
  this.vid_.setAttribute("style", style_);
  //alert(this.vid_.outerHTML)
  container.appendChild(this.vid_);
  this.vid_.onclick = function(event){
      if(event.ctrlKey && event.altKey){
        var div_=document.createElement("div");
        var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
        style_+="width:400px;height:400px";
        div_.setAttribute("style", style_);
        this.container.appendChild(div_);
        var input_file_=document.createElement("input");
        input_file_.setAttribute("id", "input_file_"+obj_number);
        input_file_.setAttribute("type", "file");
        input_file_.setAttribute("class", "filestyle");
        input_file_.setAttribute("data-icon", "true");
        input_file_.setAttribute("data-buttonText", "Brows");
        input_file_.setAttribute("file_name", dic["properties"]["src"]);
        input_file_.setAttribute("name", "driver_file");
        div_.appendChild(input_file_);
        input_file_.atm=this.my_creator_obj.parent.atm
        input_file_.dic=this.dic;
        input_file_.div_=div_;

        input_file_.onchange = function(event){
                var e = event.target;
                var filename = "";try{filename = e.getAttribute("file_name");} catch(er){}
                var file = this.files[0];
                var xhr = new XMLHttpRequest();
                try{xhr.dic=this.dic} catch(er){alert(er)}
                (xhr.upload || xhr).addEventListener('progress', function(e) {
                    try{var done = e.position || e.loaded
                    var total = e.totalSize || e.total;
                    console.log('xhr progress: ' + Math.round(done/total*100) + '%');} catch(er) {}
                });
                xhr.div_=this.div_;
                xhr.addEventListener('load', function(e) {
                    //data = JSON.parse(this.responseText);
                    location.reload();
                    this.div_.outerHTML="";
                    try{var s="zz="+this.dic["functions"]["on_loaded"];eval(s);zz(event)} catch(er){}
                });

                //alert(this.atm.upload_file_link_)
                xhr.open('post', this.atm.upload_file_link_, true);
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                var fd = new FormData();

                fd.append("user_id", "");
                fd.append("sheet_name", "");
                fd.append("entity_name", "");

                fd.append("filename", filename);
                fd.append("drive_file", file);
                fd.append("app", this.atm.my_app);
                fd.append("obj_name", this.dic["properties"]["obj_name"]);
                fd.append("function_name", this.dic["properties"]["function_name"]);
                fd.append("topic_id", "video");
                fd.append("folder_type", "media");
                fd.append("dimensions", "");
                fd.append("fields", "");
                fd.append("fact_model_field", "");
                //--
                fd.append("model_name", "");
                //--
                xhr.send(fd);
                try{var s="zz="+dic["functions"]["onchange"];eval(s);zz(event)} catch(er){}
      }
      }
  }
    // alert("90321-2 "+JSON.stringify(dic["functions"]));
  for(var f in dic["functions"])
  {if((f!="onchange") && (f!="on_loaded")){var s="this.input_file_."+f+"="+dic["functions"][f];eval(s);}}
}

// Radio --
function acActivityCreator(parent){this.parent=parent;
   var dic=this.parent.data;
   this.number_of_radios=10;if(dic["properties"]["number_of_radios"]){this.number_of_radios=dic["properties"]["number_of_radios"]}
   this.id_width = 50;
   this.name_width = 150;
   this.score_width = this.number_of_radios*47;
   this.table_width = 1*this.id_width + 1*this.name_width+1*this.score_width;
   this.field = "specialty";
   this.field_value = 45;
   this.sf_field = "test";
   this.sf_field_value = 50;
   this.sub_field_name="value";
}


acActivityCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  var container = document.getElementById("content_"+dic["container_id"]);
  var obj_number = dic["properties"]["obj_number"];
  if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
  //alert(JSON.stringify(dic));
  this.parent.data=dic;
  //--
  this.number_of_rows=dic["properties"]["number_of_rows"];
  this.model=dic["properties"]["table"];
  this.parent_model=dic["properties"]["parent_table"];
  this.table_=document.createElement("table");
  this.table_.addEventListener("change", function(event){
    var e=event.target
    e.setAttribute("model", this.getAttribute("model"))
    e.setAttribute("parent_model", this.getAttribute("parent_model"))
    e.setAttribute("content_type", "activity")
   })
  this.table_.setAttribute("class", dic["properties"]["table_class"]);
  this.table_.setAttribute("model", this.model);
  this.table_.setAttribute("parent_model", this.parent_model);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", dic["properties"]["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);
  this.table_.setAttribute("style", "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+this.table_width+"px");
  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);
  var tr_h=document.createElement("tr");
  tr_h.my_creator_obj=this;
  tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(tr_h);
    var th_an=document.createElement("th");th_an.innerHTML="ID";
    th_an.setAttribute("style","width:"+this.id_width+"px;border-top-left-radius: 15px");tr_h.appendChild(th_an);
    var th_a=document.createElement("th");th_a.innerHTML="Name";
    th_a.setAttribute("style","width:"+this.name_width+"px;");tr_h.appendChild(th_a);
    var th_m=document.createElement("th");th_m.innerHTML="Score";
    th_m.setAttribute("style","width:"+this.score_width+"px;border-top-right-radius: 15px");
    tr_h.appendChild(th_m);
  this.tbody=document.createElement("tbody");
  this.tbody.setAttribute("id", "tbody_"+dic["properties"]["obj_number"]);
  this.tbody.setAttribute("style", "display: block;");
  this.table_.appendChild(this.tbody);
  this.table_.my_creator_obj=this;

  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in dic["properties"]){this.table_.setAttribute(s, dic["properties"][s])}
   else{this.table_.setAttribute(s, "")}}

  container.appendChild(this.table_);
  // this.table_.onclick=this.row_click;
  for(f in dic["functions"]){if(f!="onclick"){var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}
}


acActivityCreator.prototype.set_members = function(members_dic)
{
 var dic = this.parent.data
 this.members_dic = members_dic
 //var obj_number_ = dic["obj_number"]
 var obj_number_ = 50;
 this.tbody.innerHTML=""
 for(var m in this.members_dic)
 {
  var tr=document.createElement("tr");this.tbody.appendChild(tr);
  var tdan=document.createElement("td");tdan.setAttribute("style","width:"+this.id_width+"px");tdan.innerHTML=m;tr.appendChild(tdan);
  var tda=document.createElement("td");tda.setAttribute("style","width:"+this.name_width+"px");tda.innerHTML=this.members_dic[m];tr.appendChild(tda);
  var tdi=document.createElement("td");tdi.setAttribute("style","width:"+this.score_width+"px");tr.appendChild(tdi);
     var div_s=document.createElement("div");
     div_s.setAttribute("record_id", "new");
     div_s.setAttribute("field", this.sf_field);
     for(var j=1; j<=this.number_of_radios; j++)
     {
        var label_=document.createElement("label");
        label_.setAttribute("class", "radio-inline");
        var span_=document.createElement("span");span_.innerHTML=j;label_.appendChild(span_);
        var input_field_=document.createElement("input");input_field_.setAttribute("type", "radio");
        input_field_.setAttribute("container_id", dic["container_id"]);
        input_field_.setAttribute("field", this.sub_field_name);
        input_field_.setAttribute("parent_id", m);
        input_field_.setAttribute("fields_values", "obj_number"+"__"+obj_number_+"-"+this.field+"__"+this.field_value+"-"+this.sf_field+"__"+this.sf_field_value+"-"+this.sub_field_name+"__"+j);
        input_field_.setAttribute("id", m+"_"+obj_number_+"_"+this.field_value+"_"+this.sf_field_value+"_"+j);
        input_field_.setAttribute("radio_value", j);
        input_field_.setAttribute("name", "radio_"+m+"_"+obj_number_+"_"+this.field_value+"_"+this.sf_field_value);label_.appendChild(input_field_);
        var span_=document.createElement("span");span_.innerHTML="&nbsp;&nbsp;&nbsp;";label_.appendChild(span_);
        div_s.appendChild(label_);
     }
    tdi.appendChild(div_s);
  var table_id=this.parent.data["properties"]["obj_number"]
 }
}

acActivityCreator.prototype.set_data = function(parent_id, obj_nums=[])
{
 //alert(parent_id)
 //alert(JSON.stringify(obj_nums));
  var fun=function(data, obj_nums){
    try{
        if(obj_nums.constructor === Array){
         for(j in obj_nums)
         {
            var table_id=obj_nums[j]+"";
            var n_=data["account"].length;
            var s='';
            for(i=0; i<n_; i++)
            {
             try{
               var i_=document.getElementById(table_id+"_"+data["account"][i])
               i_.setAttribute("record_id", data["id"][i])
               i_.value=data["amount"][i]
               } catch(er) {}
            }
         }
        }
    }catch(er){}
  }
  this.set_accounts(null,obj_nums);

  var model_ = document.getElementById("content_"+this.parent.data["container_id"]).getAttribute("table");
  var dic_={"model":this.model, "number_of_rows":1000, "fields":["account", "amount", "comment"],
             "parent_model":model_ , "parent_id": parent_id, filters:{}, order_by:{}}

  if(obj_nums.length<1){obj_nums.push(this.parent.data["properties"]["obj_number"])}
  this.parent.atm.get_data(call_back_fun=fun, dic_, obj_nums)
}

acActivityCreator.prototype.row_click = function(event)
{
 var e=event.target;
 while(e.tagName!="TR"){e=e.parentNode;}
 //alert("1: "+e.outerHTML)
 //alert(this.my_creator_obj.table_.outerHTML)
 var n_row=e.getAttribute("row")
 if(n_row==null){return;}

 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);
 //alert(container_id)

 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event)} catch(er)
 {//alert("er9013: "+er)
 }

}

// acRadio --
function acRadioCreator(parent){this.parent=parent;this.structure_dic=null;}

acRadioCreator.prototype.create_obj = function()
{
  //try{alert("9088-3 "+this.parent.atm);} catch(er){alert("13: "+er)}
  //try{alert("9088-4 "+this.parent.atm.general)} catch(er){alert("14: "+er)}
  //try{alert("9088-5 "+JSON.stringify(this.parent.atm.general.specialty))} catch(er){alert("15: "+er)}
  this.fields=[];
  var dic=this.parent.data;
  var container = document.getElementById("content_"+dic["container_id"]);
  //alert("90355 "+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  this.fields.push("obj_number");
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  this.main_div.setAttribute("style", style_);
  this.main_div.setAttribute("container_id", dic["container_id"]);
  container.appendChild(this.main_div);
  //--
  var general_dic_name = dic["properties"]["setup_dictionary"]
  if(general_dic_name==null || general_dic_name==""){

    //alert("i am not defined write code to set property")
    this.main_div.innerHTML="Radio Pluging"

  } else {
      eval("var structure_dic=this.parent.atm.general."+general_dic_name);
      this.structure_dic = structure_dic;
      // try{alert("9088-6 "+JSON.stringify(structure_dic))} catch(er){alert("15: "+er)}
      var field=structure_dic["field"]; this.fields.push(field);
      var sd_data=structure_dic["data"]
      //try{alert("9088-4 "+JSON.stringify(this.fields))} catch(er){alert("15: "+er)}
      var ny_=dic["properties"]["y"]
      for(var h in sd_data)
      {
        div_=document.createElement("div");
        this.main_div.appendChild(div_);
        div_.setAttribute("id", obj_number+"_"+h);
        var h3_=document.createElement("h3");h3_.innerHTML= sd_data[h]["title"]+"&nbsp;&nbsp;";
        div_.appendChild(h3_);
        var n_radios=sd_data[h]["range"];if(n_radios==null || n_radios==""){number_of_radios=5}
        //--
        var sub_field = sd_data[h]["sub_field"];
        var sf_field = sub_field["field"];if(!this.fields.includes(sf_field)){this.fields.push(sf_field)};
        var sf_data = sub_field["data"];
        var sub_field_name = sf_data["field"];
        if(!this.fields.includes(sub_field_name)){this.fields.push(sub_field_name)};
      //try{alert("9088-5 "+JSON.stringify(this.fields))} catch(er){alert("15: "+er)}
        var sf_data = sf_data["data"];
        var l_=0; var max_=0;for(var z in sf_data){l_+=1;var l=sf_data[z].length; if(l>max_){max_=l}}; max_*=10
        try{
          //try{alert("9088-1 "+JSON.stringify(sf_data))} catch(er){alert("15: "+er)}
           for(var z in sf_data)
           {
             var div_s=document.createElement("div");
             div_s.setAttribute("record_id", "new");
             div_s.setAttribute("field", sf_field);
             var span_=document.createElement("span");span_.innerHTML=sf_data[z]+"&nbsp;:&nbsp;";div_s.appendChild(span_);
             span_.setAttribute("style", "display:inline;float:left;width:"+max_+"px;text-align:right");
             for(var j=1; j<=n_radios; j++)
             {
                var label_=document.createElement("label");
                label_.setAttribute("class", "radio-inline");
                var span_=document.createElement("span");span_.innerHTML=j;label_.appendChild(span_);
                var input_file_=document.createElement("input");input_file_.setAttribute("type", "radio");
                input_file_.setAttribute("container_id", dic["container_id"]);
                input_file_.setAttribute("field", sub_field_name);
                input_file_.setAttribute("fields_values", "obj_number"+"__"+obj_number+"-"+field+"__"+h+"-"+sf_field+"__"+z+"-"+sub_field_name+"__"+j);
                input_file_.setAttribute("id", obj_number+"_"+h+"_"+z+"_"+j);

                input_file_.setAttribute("radio_value", j);

                input_file_.setAttribute("name", "radio_"+obj_number+"_"+h+"_"+z);label_.appendChild(input_file_);
                var span_=document.createElement("span");span_.innerHTML="&nbsp;&nbsp;&nbsp;";label_.appendChild(span_);
                div_s.appendChild(label_);
             }
            div_.appendChild(div_s);
           }
         } catch (er) {alert(er)}
      }
 }

// this.main_div.onclick=this.on_radio_click;
// for(f in dic["functions"]){if(f!="onclick"){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}}
 for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
}

//acRadioCreator.prototype.on_radio_click = function(event)
//{
// try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event);} catch(er){}
//}

acRadioCreator.prototype.clean_data = function()
{
  var dic=this.parent.data;
  var obj_number = dic["properties"]["obj_number"]
  //--
  var general_dic_name = dic["properties"]["setup_dictionary"]
  eval("var structure_dic=this.parent.atm.general."+general_dic_name);
  this.structure_dic = structure_dic;
  var sd_data=structure_dic["data"]
  for(var h in sd_data)
  {
    var n_radios=sd_data[h]["range"];if(n_radios==null || n_radios==""){number_of_radios=5}
    var sf_data = sd_data[h]["sub_field"]["data"]["data"];
    try{
       for(var z in sf_data)
       {
         for(var j=1; j<=n_radios; j++)
         {
            s_id=obj_number+"_"+h+"_"+z+"_"+j;var o=getEBI(s_id);o.checked=false;
            var p=o.parentNode.parentNode;
            p.setAttribute("record_id", "new")
         }
       }
     } catch (er) {alert(er)}
  }
}

acRadioCreator.prototype.set_data = function(ll=null)
{
  this.clean_data();if(ll!=null){for(var i in ll){get_creator(ll[i]).clean_data()}}
  var dic=this.parent.data;
  //alert("acRadioCreator 90441-1  "+JSON.stringify(dic));
  var container = document.getElementById("content_"+dic["container_id"]);
     var model_=container.my_creator_obj.link_dic["properties"]["table"];
     var parent_model_=container.my_creator_obj.link_dic["properties"]["parent_table"];
     var parent_id_=container.getAttribute("parent_id");
     var dic_={"model":model_, "parent_model":parent_model_, "parent_id": parent_id_, "number_of_rows": "1000",
               "fields": this.fields, "filters":{}, "order_by": {}}

    //alert("90446-1  "+JSON.stringify(dic_));
    //alert("90441-2  "+JSON.stringify(this.fields));

  var fun=function(data, this_obj){
    var dic=this_obj.parent.data;
    var obj_number = dic["properties"]["obj_number"]
    var n_=data[this_obj.fields[0]].length;
    //alert(n_)
    if(n_<1){return}
    //alert("9081-22 acRadioCreator.prototype.set_data\n data: "+JSON.stringify(data));
    for(var j=0; j<n_; j++)
    {
     var s_id=data[this_obj.fields[0]][j]+"_"+data[this_obj.fields[1]][j]+"_"+data[this_obj.fields[2]][j]+"_"+data[this_obj.fields[3]][j]
     getEBI(s_id).checked=true;
     getEBI(s_id).parentNode.parentNode.setAttribute("record_id", data["id"][j])
    }
  }
  //  alert("901210: "+JSON.stringify(dic__));
  this.parent.atm.get_data(call_back_fun=fun, dic_, this)
}


// acTest --
function acTestCreator(parent){this.parent=parent;this.structure_dic=null;this.test_list=[];this.fields=[];
                   this.members_list={"test_number":[], "record_id":[]};
}

acTestCreator.prototype.create_obj = function()
{
  var ff = function(div_, dic_, dic, nlt, ll){
      table_=document.createElement("table");
      t_body_=document.createElement("tbody");
      table_.appendChild(t_body_);
      div_.appendChild(table_);
      for(var l in dic_){f(e_obj=table_, l, dic_[l], dic, nlt, ll)}
  }
  var f = function(e_obj, l, dic_, dic, nlt, ll)
  {
    //alert("\n" + JSON.stringify(dic))
    //alert("\n" +l+": "+ JSON.stringify(dic_))
    //this_obj.test_dic["test_number"].push(l)
    //this_obj.test_dic["test_name"].push(dic_["title"])
   ll.push(l)
   var container_id = dic["container_id"]
   var table = dic["properties"]["table"]
   var field = dic["properties"]["field"]
   var value_field = dic["properties"]["value_field"]
   var s = '<input id="test_'+l+'" my_creator_number="'+dic["properties"]["obj_number"]+'"  type="checkbox" type_="tests" container_id="'+container_id
   s+='" field="'+field+'" value_field="'+value_field+'" table="'+table+'" record_id="new" test_number="'+l+'">'
   //var s = '<label class="switch"><input type="checkbox" checked><span class="slider"></span></label><br><br>'
   e_obj.innerHTML+="<tr><td style='width:"+nlt+"%'>"+dic_["title"]+"</td><td>"+s+"</td></tr>"
  }
  var dic=this.parent.data;
  var field=dic["properties"]["field"]; this.fields.push(field);
  var value_field=dic["properties"]["value_field"]; this.fields.push(value_field);

    //alert("90441-200  "+JSON.stringify(this.fields));

  var width_=dic["properties"]["width"]; if(width_==null || width_==""){width_=80};
  var height_=dic["properties"]["height"]; if(height_==null || height_==""){height_=300};
  var container = document.getElementById("content_"+dic["container_id"]);
  //alert("90355-23 acTestDesignerCreator.prototype.create_obj\n"+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.my_creator_obj.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"%"
  this.main_div.setAttribute("style", style_);
  this.main_div.setAttribute("class", "row");
  this.main_div.setAttribute("container_id", dic["container_id"]);

  this.main_div.addEventListener("click", function(event){
      var e=event.target;
      //alert(e.outerHTML)
      //alert("90765-57\n"+JSON.stringify(this.dic));
      var etype_=e.getAttribute("type")
      if(etype_=="checkbox"){
         //alert(e.outerHTML)
         //alert(e.checked)
          var etype_=e.getAttribute("type")
          if(etype_=="checkbox"){
             //alert("test66")
             this.my_creator_obj.obj_was_clicked(e)
          }
      }
  })

  container.appendChild(this.main_div);
  //--
  var general_dic_name = dic["properties"]["setup_dictionary"]
  if(general_dic_name==null || general_dic_name==""){
    //alert("i am not defined write code to set property")
    this.main_div.innerHTML="Test Pluging"
  } else {
      try{eval("var structure_dic=this.parent.atm.general."+general_dic_name)} catch(er){alert("1567-98: "+er)}
      this.structure_dic = structure_dic;
      //try{alert("9088-63 "+JSON.stringify(structure_dic))} catch(er){alert("15: "+er)}

      var nl = Math.round(10000/Object.keys(structure_dic).length)/100
      var nlt = 90
      for(var h in structure_dic)
      {
        //alert(h + "\n" + JSON.stringify(structure_dic[h]))
          div_=document.createElement("div");
          div_.setAttribute("id", obj_number+"_"+h);
          var style_ = "float:left;width:"+nl+"%;padding:10px;height:"+height_+"px;border-radius:25px;background-color:"
          var color_ = structure_dic[h]["color"];
          style_ += color_+";padding: 10px;border: 2px solid #73AD21;";
          div_.setAttribute("style", style_);
          h_=document.createElement("h3");
          div_.appendChild(h_);
          var img_=structure_dic[h]["img"];
          var s_img="";if(img_!=null && img_!=""){s_img="<img style='width:25px;height:25px;' src='/media/training/images/"+structure_dic[h]["img"]+"'>&nbsp;&nbsp;&nbsp;"}
          h_.innerHTML = s_img+structure_dic[h]["title"]+"<hr/>"
          this.main_div.appendChild(div_);
          var dic_tests=structure_dic[h]["sub_tests"]
          if(dic_tests!=null)
          {
            if(dic_tests["type"]==2){
              for(var g in dic_tests["data"])
              {
                var title_ = dic_tests["data"][g]["title"]
                var color_ = dic_tests["data"][g]["color"]
                div_.innerHTML += "<h3 style='text-align:center;width:100%;color:"+color_+"'><u>"+title_+"</u></h3>"
                var dic_=dic_tests["data"][g]["sub_tests"]["data"]
                ff(div_, dic_, dic, nlt, this.test_list)
                //f1(dic_, this.test_list)
              }
            } else if(dic_tests["type"]==3)
            {
              ff(div_, dic_tests["data"], dic, nlt, this.test_list);
              //f1(dic_tests["data"], this.test_list)
            }
          }
      }
      //alert(JSON.stringify(this.test_list));
  }

  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
}

acTestCreator.prototype.clean_data = function()
{for(var j in this.test_list){var id="test_"+this.test_list[j];getEBI(id).checked=false;}}

acTestCreator.prototype.obj_was_clicked = function(e)
{
  //alert("901456\nacTestCreator.prototype.obj_was_clicked\n "+e.outerHTML);
}

acTestCreator.prototype.set_data = function(record_id, ll=null)
{
  this.clean_data();if(ll!=null){for(var i in ll){get_creator(ll[i]).clean_data()}}
  var dic=this.parent.data;var pp_=dic["properties"];
  this.recording_tests_number=pp_["recording_tests_number"];
  //alert("acRadioCreator 90446-16  "+JSON.stringify(dic));
  var model_=pp_["table"]
  var container = document.getElementById("content_"+dic["container_id"]);
  var parent_model_=container.my_creator_obj.link_dic["properties"]["table"];
     var dic_={"model":model_, "parent_model":parent_model_, "parent_id": record_id, "number_of_rows": "1000",
               "fields": this.fields,
               "filters":{}, "order_by": {}}
    //alert("90447-150  "+JSON.stringify(dic_));
  var fun=function(data, this_obj){
    //alert("943010: "+JSON.stringify(data));
    this_obj.members_list={"test_number":[], "record_id":[]}
    var n_=data[this_obj.fields[0]].length;if(n_<1){return}
    for(var j=0; j<n_; j++)
    {
     var s_id="test_"+data["test_number"][j];var n=1*data["value"][j];
     s_id=getEBI(s_id);if(n==1){s_id.checked=true}else{s_id.checked=false}
     s_id.setAttribute("record_id",data["id"][j])
     //alert(data["id"][j])
     //alert("90447-150  "+JSON.stringify(this_obj.members_list));
     this_obj.members_list["test_number"].push(data["test_number"][j])
     this_obj.members_list["record_id"].push(data["id"][j])
    }
     try{
       get_creator(this_obj.recording_tests_number).set_obj_structure(group_list=null, test_list=this_obj.members_list)
     } catch(er){}
  }
  this.parent.atm.get_data(call_back_fun=fun, dic_, this)
}

acTestCreator.prototype.on_record_created_deleted = function(e,action="created")
{
   //alert("90120-20\nacTestCreator.prototype.on_record_created_deleted\n action: "+action+" \n\n"+e.outerHTML);
   try{
    var record_id=e.getAttribute("record_id")
    if(action=="deleted"){e.setAttribute("record_id", "new")}
   } catch(er){alert(er)}
}


// acVariable --
function acVariableCreator(parent){this.parent=parent;this.structure_dic=null;this.variable_list=[];this.fields=[];
                   this.members_list={"variable_number":[], "record_id":[]};
    // alert("acVariableCreator=\n" + JSON.stringify(this.parent.data))
}

acVariableCreator.prototype.create_obj = function()
{
  var ff = function(div_, dic_, dic, nlt, ll){
      table_=document.createElement("table");
      t_body_=document.createElement("tbody");
      table_.appendChild(t_body_);
      div_.appendChild(table_);
      //alert(div_.outerHTML)
      for(var l in dic_){f(e_obj=table_, l, dic_[l], dic, nlt, ll)}
  }
  var f = function(e_obj, l, dic_, dic, nlt, ll)
  {
    //alert("\n" + JSON.stringify(dic))
    //alert("\n" +l+": "+ JSON.stringify(dic_))
   ll.push(l)
   var container_id = dic["container_id"]
   var table = dic["properties"]["table"]
   var field = dic["properties"]["field"]
   var value_field = dic["properties"]["value_field"]
   var s = '<input id="variable_'+l+'" my_creator_number="'+dic["properties"]["obj_number"]+'" type="text" size="2" type_="variables" container_id="'+container_id
   s+='" field="'+field+'" value_field="'+value_field+'" table="'+table+'" record_id="new" test_number="'+l+'">'
   //var s = '<label class="switch"><input type="text" checked><span class="slider"></span></label><br><br>'
   s="<tr><td style='width:"+nlt+"%'>"+dic_["title"]+"</td><td>"+s+"</td></tr>"
   //alert("s: "+s)
   e_obj.innerHTML+=s
   //alert(e_obj.outerHTML)
  }
  var dic=this.parent.data;
    //alert("90441-200  "+JSON.stringify(dic));
  var field=dic["properties"]["field"]; this.fields.push(field);
  var value_field=dic["properties"]["value_field"]; this.fields.push(value_field);
  //alert("90441-205  "+JSON.stringify(this.fields));
  var width_=dic["properties"]["width"]; if(width_==null || width_==""){width_=80};
  var height_=dic["properties"]["height"]; if(height_==null || height_==""){height_=300};
  var container = document.getElementById("content_"+dic["container_id"]);
  //alert("90355-23 acvariableCreator.prototype.create_obj\n"+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.my_creator_obj.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"%"
  this.main_div.setAttribute("style", style_);
  this.main_div.setAttribute("class", "row");
  this.main_div.setAttribute("container_id", dic["container_id"]);
  this.main_div.addEventListener("click", function(event){
      var e=event.target;
      //alert(e.outerHTML)
      //alert("90765-57\n"+JSON.stringify(this.dic));
      var etype_=e.getAttribute("type")
      if(etype_=="checkbox"){
         //alert(e.outerHTML)
         //alert(e.checked)
          var etype_=e.getAttribute("type")
          if(etype_=="checkbox"){
             //alert("variable66")
             this.my_creator_obj.obj_was_clicked(e)
          }
      }
  })
  container.appendChild(this.main_div);
  //--
  var general_dic_name = dic["properties"]["setup_dictionary"]
  if(general_dic_name==null || general_dic_name==""){
    //alert("i am not defined write code to set property")
    this.main_div.innerHTML="variable Pluging"
  } else {
      try{eval("var structure_dic=this.parent.atm.general."+general_dic_name)} catch(er){alert("1567-98: "+er)}
      this.structure_dic = structure_dic;
      try{//alert("9088-63 "+JSON.stringify(structure_dic))
      } catch(er){alert("15: "+er)}

      var nl = Math.round(10000/Object.keys(structure_dic).length)/100
      var nlt = 80
      for(var h in structure_dic)
      {
        //alert(h + "\n" + JSON.stringify(structure_dic[h]))
          div_=document.createElement("div");
          div_.setAttribute("id", obj_number+"_"+h);
          var style_ = "float:left;width:"+nl+"%;"
          style_ += "padding-top: 0px; padding-right: 10px;padding-bottom: 0px;padding-left: 10px;"
          style_ += "height:"+height_+"px;border-radius:25px;background-color:"
          var color_ = structure_dic[h]["color"];
          style_ += color_+";border: 2px solid #73AD21;";
          div_.setAttribute("style", style_);
          h_=document.createElement("h3");

          var img_=structure_dic[h]["img"];
          var s_img="";if(img_!=null && img_!=""){s_img="<img style='width:25px;height:25px;' src='/media/training/images/"+structure_dic[h]["img"]+"'>&nbsp;&nbsp;&nbsp;"}
          h_.innerHTML = s_img+structure_dic[h]["title"]+"<hr/>"

          div_.appendChild(h_);
          this.main_div.appendChild(div_);
          var dic_variables=structure_dic[h]["sub_tests"]
          if(dic_variables!=null)
          {
            if(dic_variables["type"]==2){
              for(var g in dic_variables["data"])
              {
                var title_ = dic_variables["data"][g]["title"]
                var color_ = dic_variables["data"][g]["color"]

                div_.innerHTML += "<h3 style='text-align:center;width:100%;color:"+color_+"'><u>"+title_+"</u></h3>"
                var dic_=dic_variables["data"][g]["sub_tests"]["data"]
                ff(div_, dic_, dic, nlt, this.variable_list)
                //f1(dic_, this.variable_list)
              }
            } else if(dic_variables["type"]==3)
            {
              ff(div_, dic_variables["data"], dic, nlt, this.variable_list);
              //f1(dic_variables["data"], this.variable_list)
            }
          }
      }
      //alert(JSON.stringify(this.variable_list));
  }

  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
}

acVariableCreator.prototype.clean_data = function()
{for(var j in this.variable_list){var id="variable_"+this.variable_list[j];getEBI(id).value="";}}

acVariableCreator.prototype.obj_was_clicked = function(e)
{
  //alert("901456\nacvariableCreator.prototype.obj_was_clicked\n "+e.outerHTML);
}

acVariableCreator.prototype.set_data = function(record_id, ll=null)
{
  //alert(record_id)
  this.clean_data(); if(ll!=null){for(var i in ll){get_creator(ll[i]).clean_data()}}
  var dic=this.parent.data;
  this.recording_variables_number=dic["properties"]["recording_variables_number"];
  //alert("acVariableCreator 90446-16  "+JSON.stringify(dic));
  var model_=dic["properties"]["table"]
  var container = document.getElementById("content_"+dic["container_id"]);
  var parent_model_=container.my_creator_obj.link_dic["properties"]["table"];
     var dic_={"model":model_, "parent_model":parent_model_, "parent_id": record_id, "number_of_rows": "1000",
               "fields": this.fields,
               "filters":{}, "order_by": {}}
    //alert("90447-150  "+JSON.stringify(dic_));
  var fun=function(data, this_obj){
    //alert("943010: "+JSON.stringify(data));
    this_obj.members_list={"variable_number":[], "record_id":[]}
    var n_=data[this_obj.fields[0]].length;if(n_<1){return}
    for(var j=0; j<n_; j++)
    {
     var s_id="variable_"+data["test_number"][j];
     var n=1*data["value"][j];
     s_id=getEBI(s_id);s_id.value=n;s_id.setAttribute("record_id",data["id"][j])

     //alert(data["id"][j])

     this_obj.members_list["variable_number"].push(data["test_number"][j])

     this_obj.members_list["record_id"].push(data["id"][j])
     //alert("90447-150  "+JSON.stringify(this_obj.members_list));
    }
     //get_creator(this_obj.recording_variables_number).set_obj_structure(group_list=null, variable_list=this_obj.members_list)
  }
  this.parent.atm.get_data(call_back_fun=fun, dic_, this)
}

acVariableCreator.prototype.on_record_created_deleted = function(e,action="created")
{
   //alert("90120-20\nacvariableCreator.prototype.on_record_created_deleted\n action: "+action+" \n\n"+e.outerHTML);
   try{
    var record_id=e.getAttribute("record_id")
    if(action=="deleted"){e.setAttribute("record_id", "new")}
   } catch(er){alert(er)}
}


// acGroup --
function acGroupCreator(parent){this.parent=parent;this.fields=[];this.members_list={"entity_number":[], "record_id":[]};
                                this.active_record_id=-1;this.is_group_updated=false}

acGroupCreator.prototype.get_entity_list = function(e_dic)
{
 var dic=this.parent.data;
 //alert("90100-1\n"+JSON.stringify(dic));
 var obj_number=dic["properties"]["obj_number"]
 var container_id=dic["container_id"]
 var e=e_dic["e"];var c=e_dic["c"];
 var table=dic["properties"]["table"];
 var field=dic["properties"]["field"]
 var value_field = dic["properties"]["value_field"]
 //alert(e.outerHTML)
 if(e.getAttribute("my_members_status")=="empty")
 {
     e.setAttribute("my_members_status", "filled")
     //alert(e.getAttribute("link"))
     var dic_=eval(e.getAttribute("link"))
     //alert("901540-24 acGroupCreator\n"+JSON.stringify(dic_));
     for(var i in dic_)
     {
      var id_=i; var title_=dic_[i]["title"];
      var s='&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' //<span style="cursor: pointer">+</span>'
      s+='<input type="checkbox" my_creator_number="'+obj_number+'" container_id="'+container_id+'" id="soldier_'+id_+'" soldier_number="'+id_+'" record_id="new" value="" table="'+table
      s+='" field="'+field+'" value_field="'+value_field+'" type_="groups">'
      s+='<label>'+title_+'</label><br/>'
      c.innerHTML+=s
     }
 }
}

acGroupCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;var pp_=dic["properties"];
  var field=dic["properties"]["field"]; this.fields.push(field);
  var value_field=dic["properties"]["value_field"]; this.fields.push(value_field);
  var container = document.getElementById("content_"+dic["container_id"]);
   //alert("90100-2 acGroupCreator.prototype.create_obj\n"+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  var style_ = "position:absolute;left:"+pp_["x"]+"px;top:"+pp_["y"]+"px;"

  var width_=pp_["width"]; if(width_==null || width_==""){width_=200};style_+="width:"+width_+"px;"
  //var height_=pp_["height"]; if(height_==null || height_==""){height_=300};

  var bs_=pp_["border_style"];if(bs_!=null && bs_!=""){style_+="border-style:"+bs_+";"}
  var bw_=pp_["border_width"];if(bw_!=null && bw_!=""){style_+="border-width:"+bw_+"px;"}
  var bc_=pp_["border_color"];if(bc_!=null && bc_!=""){style_+="border-color:"+bc_+";"}
  var br_=pp_["border_radius"];if(br_!=null && br_!=""){style_+="border-radius:"+br_+"px;"}
  var c_=pp_["color"];if(c_!=null && c_!=""){style_+="color:"+c_+";"}
  var bgc_=pp_["background_color"];if(bgc_!=null && bgc_!=""){style_+="background-color:"+bgc_+";"}

//alert(style_)

  this.main_div.setAttribute("style", style_);
  this.main_div.setAttribute("class", "row");
  this.main_div.setAttribute("container_id", dic["container_id"]);

  this.main_div.addEventListener("change", function(event){
      var e=event.target;
   //alert("90100-3 change\n\n"+e.outerHTML)
  })

  this.main_div.addEventListener("click", function(event){
      var e=event.target;
      //alert("click: \n\n"+this.outerHTML)
      //alert("click: \n\n"+e.outerHTML)
      //alert("90100-4\n"+JSON.stringify(this.dic["properties"]));

      //alert(pp_["obj_number"]+"\n"+JSON.stringify(this.dic["properties"]))

      var my_class_members=e.getAttribute("my_class_members"+pp_["obj_number"])

      //alert("90142-2 acGroupCreator.prototype.create_obj\n"+JSON.stringify(this.dic));
      //alert(this.dic["properties"]["present"])
  try{
      if(my_class_members!=null){
        //alert(this.dic["properties"]["present"])
        var c = document.getElementsByClassName(my_class_members)[0]
        var etype_=e.getAttribute("type")
        if(etype_=="checkbox"){
        try{
          if(this.dic["properties"]["present"]=="detail")
          {
             //alert(e.outerHTML)
             //alert(e.checked)
             var cc=c.getElementsByTagName("input");
             var ec=new Event("change", {bubbles: true});
             var n_=0
             for(k in cc)
             {
              cc[k].checked=e.checked;
             }
             for(k in cc)
             {
             //alert(cc[k].outerHTML)

              cc[k].dispatchEvent(ec)
             }
          }
        } catch(er){"Error 550: "+er}
        } else if (etype_=="span"){
        //alert(999)
        //alert(c.outerHTML)
        //alert(c.style.display)

          if(c.style.display=="block"){c.style.display="none"}else{c.style.display="block";
          if(this.dic["properties"]["present"]=="detail")
          {
            this.my_creator_obj.get_entity_list({"e":e,"c":c})}
          }
        }

      } else{
        var etype_=e.getAttribute("type")
        if(etype_=="checkbox"){
         //alert("group55")
         this.my_creator_obj.obj_was_clicked(e)
        }
      }

} catch(er){alert("Error 555\n"+er)}

  })
  container.appendChild(this.main_div);
  //--
  var general_dic_name=dic["properties"]["setup_dictionary"]
  if(general_dic_name==null || general_dic_name==""){this.main_div.innerHTML="Group Pluging"}
  else {this.main_div.innerHTML="Group Pluging working"}
  //--
  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
 // alert(this.main_div.outerHTML)
}

acGroupCreator.prototype.obj_was_clicked = function(e)
{
  //alert("90100-10\nacGroupCreator.prototype.obj_was_clicked\n "+e.outerHTML);
}

acGroupCreator.prototype.open_close_lists = function(n=0)
{
  var dic=this.parent.data;var pp_=dic["properties"]
  var ll=atm.general.get_platoons_ids();ll=ll[n]
  //alert("90667-77\n"+JSON.stringify(ll));
  var ec=new Event("click", {bubbles: true});
  for(l in ll){try{getEBI(ll[l]+pp_["obj_number"]).dispatchEvent(ec)} catch(er){alert(er)}}
}

acGroupCreator.prototype.set_obj_structure = function(display_="none", nlimit=-1)
{
  //alert("90105-55\n"+nlimit+"\n"+JSON.stringify(this.parent.data["properties"]["obj_number"]));
  var pp_=this.parent.data["properties"];
  var f=function(s_link, s_title, dic, d_div, n=0, n_limit_=-1, spaces="")
  {
     var spaces_=spaces+"&nbsp;&nbsp;&nbsp;"; var n_=n+1;
     for(k in dic)
     {
        //alert(k+"\n"+dic[k]["title"])
        s_link_=s_link+"["+k+"]['data']";var title = dic[k]["title"];
//1111111111        var title_=title.replace(/ /g, '').toLowerCase();
        s_title+=title.replace(/ /g, '').toLowerCase();
        var class_="group_"+pp_["obj_number"]+s_title;var id_="id_"+s_title+pp_["obj_number"];var sid_="sid_"+s_title+pp_["obj_number"]
        //&#45;&#45;
        var s=spaces_+'<span type="span" style="cursor: pointer" id="'+sid_+'"'
        if(n_>n_limit_)
        {s+='my_members_status="empty" link="'+s_link_+'"'}
        s+='my_class_members'+pp_["obj_number"]+'="'+class_+'">+</span>'
        s+='<input type="checkbox" id="'+id_+'" '
        s+='my_class_members'+pp_["obj_number"]+'="'+class_+'" my_class_members="'+class_+'" style="cursor:pointer;"><label>'+title+'</label><br/>'
        //alert(s)
        d_div.innerHTML+=s;
        var d_div_=document.createElement("div");
        d_div_.setAttribute("id", "div_"+id_);d_div_.setAttribute("class", class_);
        //var display_='block';
        //var display_='none';
        if(n_>n_limit_){display_='none'}
        d_div_.setAttribute("style", 'display:'+display_);
        d_div.appendChild(d_div_);
        if(n_<=n_limit_){
         f(s_link_, s_title, dic[k]["data"], d_div_, n_, n_limit_, spaces_)}
     }
  }

  var dic=this.parent.data;
  //alert("acRadioCreator 90446-16  "+JSON.stringify(dic));
  try{
       var setup_dictionary = dic["properties"]["setup_dictionary"];
       this.units_structure = eval("this.parent.atm.general."+setup_dictionary)
       var general_entity_name = dic["properties"]["entity"];
       this.entity_list = eval("this.parent.atm.general."+general_entity_name+"_list")
       this.entity_dic = eval("this.parent.atm.general."+general_entity_name+"_dic")
       //alert("90000-010  "+JSON.stringify(this.entity_dic));
       //alert("90000-020  "+JSON.stringify(this.units_structure));
       var s_link="this.units_structure"; var s_title=""
       this.main_div.innerHTML="";
       //alert(nlimit)
       //alert("acGroupCreator.prototype.set_obj_structure 90446-17\n"+JSON.stringify(this.units_structure));
       f(s_link, s_title, this.units_structure, this.main_div, 0, n_limit_=nlimit)
     } catch(er){alert("1155-515: "+er)}
 //  alert("done acGroupCreator.prototype.set_obj_structure\n\n"+this.main_div.outerHTML)
}

acGroupCreator.prototype.set_data = function(record_id=null, ll=null)
{
  if(this.active_record_id==record_id){return}
  this.active_record_id=record_id;
  // --
   try{this.open_close_lists(0)} catch(er){}
  // --
  var dic=this.parent.data;var pp_=dic["properties"];
    //alert("90333-103\n"+JSON.stringify(dic));

  this.recording_tests_number=pp_["recording_tests_number"];

  var model_=dic["properties"]["table"]
  var container = document.getElementById("content_"+dic["container_id"]);
  var parent_model_=container.my_creator_obj.link_dic["properties"]["table"];

  var dic_={"model":model_,"parent_model":parent_model_,"parent_id":record_id,"number_of_rows":"10000",
            "fields":this.fields,"filters":{},"order_by":{}}

  //alert("90000-099\n"+JSON.stringify(dic_));

  var fun=function(data, this_obj){
    //alert("90000-100\n"+JSON.stringify(data));

    this_obj.members_list={"entity_number":[], "record_id":[]}
    var n_=data[this_obj.fields[0]].length;
    if(n_<1){return}

    //alert(555555555)
    //alert(this_obj.fields[0])
    //alert("90000-100\n"+JSON.stringify(data["soldier_number"]));

    for(var j=0; j<n_; j++)
    {
      try{var s_id_="soldier_"+data["soldier_number"][j];
      var n=1*data["value"][j];
      var s_id=getEBI(s_id_);
      //alert(s_id_)

      if(s_id==null){alert("Error_22-22: "+s_id_);continue}

      //alert(s_id.outerHTML)
         if(n==1){s_id.checked=true}else{s_id.checked=false};

         s_id.setAttribute("record_id",data["id"][j])
         this_obj.members_list["entity_number"].push(data["soldier_number"][j])
         this_obj.members_list["record_id"].push(data["id"][j])
     } catch(er){alert(er)}}

    //alert("90333-103\n"+JSON.stringify(this_obj.members_list));

    try{
      var g = get_creator(this_obj.recording_tests_number)
      g.set_obj_structure(group_list=this_obj.members_list, test_list=null)
    } catch(er){alert("Error 200-200-1 "+er)}

    try{
      get_creator(this_obj.recording_tests_number).set_data(record_id)
      } catch(er){alert("Error 200-200-2 "+er)}
    }

  //alert("90555-55\n"+JSON.stringify(dic_));
  this.parent.atm.get_data(call_back_fun=fun, dic_, this)
  // --)
   try{this.open_close_lists(0);this.open_close_lists(2)} catch(er){}
  // --
}

acGroupCreator.prototype.on_record_created_deleted = function(e,action="created")
{
   //alert("90100-20\nacGroupCreator.prototype.obj_was_clicked\n "+e.outerHTML);
   var record_id=e.getAttribute("record_id")
   var soldier_number=e.getAttribute("soldier_number")
   var n__=this.members_list["record_id"].length
   if(action=="created"){
      if(n__!=0){
         if(!(record_id in this.members_list["record_id"])){
          this.members_list["entity_number"].push(soldier_number);this.members_list["record_id"].push(record_id);
         }
      } else{
        if(!(soldier_number in this.members_list["entity_number"])){this.members_list["entity_number"].push(soldier_number)}
      }
   } else
   {
    if(n__!=0){
     for(var i=0;i<n__;i++){if(this.members_list["record_id"][i]==record_id) {this.members_list["record_id"].splice(i, 1);break;}}
    } else{
     var n__=this.members_list["entity_number"].length;
     for(var i=0;i<n__;i++){if(this.members_list["entity_number"][i] == soldier_number) {this.members_list["entity_number"].splice(i, 1);break;}}
    }
     //alert(record_id+"\n\n90355-1234-3 acGroupCreator.prototype.on_record_created_deleted\n"+JSON.stringify(this.members_list));
  }

//  alert(this.recording_tests_number)
//  alert(get_creator(this.recording_tests_number))
//  alert(get_creator(this.recording_tests_number).create_html)
//  alert(get_creator(this.recording_tests_number).set_obj_structure)
//  get_creator(this.recording_tests_number).set_obj_structure(group_list=this.members_list, test_list=null)
}


// acRecordingTests --
function acRecordingTestsCreator(parent){
 this.parent=parent;this.vertical_field=null;this.horizontal_field=null;this.fields=[];
}

acRecordingTestsCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //alert("90355-223 acRecordingTestsCreator.prototype.create_obj\\nn"+JSON.stringify(dic));
  var field=dic["properties"]["field"]; this.fields.push(field);
  var vertical_field=dic["properties"]["vertical_field"]; this.fields.push(vertical_field);
  var horizontal_field=dic["properties"]["horizontal_field"]; this.fields.push(horizontal_field);

  var width_=dic["properties"]["width"]; if(width_==null || width_==""){width_=80};
  var height_=dic["properties"]["height"]; if(height_==null || height_==""){height_=300};
  var container = document.getElementById("content_"+dic["container_id"]);
  // alert("90355-213 acGroupCreator.prototype.create_obj\n"+JSON.stringify(dic));
  var obj_number = dic["properties"]["obj_number"]
  this.main_div=document.createElement("div");
  this.main_div.my_creator_obj=this;
  this.main_div.dic=dic;
  this.main_div.setAttribute("id", obj_number);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.main_div.setAttribute("type", dic["element_name"]);
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"%"
  this.main_div.setAttribute("style", style_);

  this.main_div.setAttribute("container_id", dic["container_id"]);
  this.main_div.addEventListener("click", function(event){
      var e=event.target;
      //alert(e.outerHTML)
      //alert("90777-77\n"+JSON.stringify(this.dic));
  })

  this.main_div.addEventListener("change", function(event){
      var e=event.target;
      //if(e.value==""){e.value=0.00}
   //alert("90150-3 change\n\n"+e.outerHTML)
  })

  container.appendChild(this.main_div);
  //--
  this.main_div.innerHTML="RecordingTests Pluging"
  //--
  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
}

acRecordingTestsCreator.prototype.set_obj_structure = function(group_list=null, test_list=null)
{
  //alert("acRecordingTests 90000-033\n\n"+JSON.stringify(group_list));
  //--
  this.test_dic = eval("this.parent.atm.general.test_dic")

  //alert("this.test_dic 90000-035\n\n"+JSON.stringify(this.test_dic));
  //--
  var dic=this.parent.data;
  var width_=80;var userid_width_=dic["properties"]["userid_width"];
  var name_width_=dic["properties"]["name_width"];
 if(test_list!=null){this.horizontal_field_list=test_list}
 if(group_list!=null){this.vertical_field_list=group_list}

 if(this.horizontal_field_list!=null && this.vertical_field_list!=null)
 {
  //alert("acRecordingTests 90000-033\n\n"+JSON.stringify(this.vertical_field_list)+"\n"+JSON.stringify(this.horizontal_field_list));
  //alert("acRecordingTests 90000-034\n\n"+JSON.stringify(this.test_dic));
   var dic=this.parent.data;
   if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
   var general_entity_name = dic["properties"]["entity"];
   this.entity_list = eval("this.parent.atm.general."+general_entity_name+"_list")
   //alert("this.parent.atm.general."+general_entity_name+"_dic")
   this.entity_dic = eval("this.parent.atm.general."+general_entity_name+"_dic")
   var container_id=dic["container_id"];
   var field=dic["properties"]["field"];
   this.horizontal_field=dic["properties"]["horizontal_field"];
   this.vertical_field=dic["properties"]["vertical_field"];
   var table=dic["properties"]["table"];
   this.main_div.innerHTML="";
   this.table_=document.createElement("table");this.main_div.appendChild(this.table_);
   this.table_.setAttribute("class", dic["properties"]["table_class"]);
   this.thead_=document.createElement("thead");this.table_.appendChild(this.thead_);
   this.tr_=document.createElement("tr");this.thead_.appendChild(this.tr_);
    this.tr_.my_creator_obj=this;
   this.tr_.onclick=function(event){
    var e=event.target;
    var sort_by=e.getAttribute("sort_by");var sort_ascending=e.getAttribute("sort_ascending");
    if(sort_by!=null)
    {
      var new_sort=[];var kk=[...this.my_creator_obj.matching_set[sort_by]]
      if(sort_ascending=="true"){kk.sort();e.setAttribute("sort_ascending","false")}else{kk.sort().reverse();e.setAttribute("sort_ascending","true")}
      for(var j in kk)
      {new_sort.push(this.my_creator_obj.matching_set["record_id"][this.my_creator_obj.matching_set[sort_by].indexOf(kk[j])])}
      //alert(JSON.stringify(this.my_creator_obj.matching_set["record_id"])+"\n\n"+JSON.stringify(new_sort));
      this.my_creator_obj.tbody_.innerHTML=this.my_creator_obj.create_sorted_structure(v_list=new_sort, width_, field, container_id, table)

 try{if(this.my_creator_obj.data != null){this.my_creator_obj.fill_up_data(this.my_creator_obj.data, this.my_creator_obj)}} catch(er){alert(er)}

    }
   }

   var th_=document.createElement("th");this.tr_.appendChild(th_);th_.innerHTML="User id";
   th_.setAttribute("sort_by", "userid");th_.setAttribute("sort_ascending", "true");
   th_.setAttribute("style", "width:"+userid_width_+"px;border-top-left-radius: 10px");

   var th_=document.createElement("th");this.tr_.appendChild(th_);th_.innerHTML="First Name";
   th_.setAttribute("style", "width:"+name_width_+"px");
   th_.setAttribute("sort_by", "first_name");th_.setAttribute("sort_ascending", "true");

   var th_=document.createElement("th");this.tr_.appendChild(th_);th_.innerHTML="Last Name";
   th_.setAttribute("style", "width:"+name_width_+"px");
   th_.setAttribute("sort_by", "last_name");th_.setAttribute("sort_ascending", "true");

   var n_=this.horizontal_field_list["record_id"].length
   for(var i in this.horizontal_field_list["record_id"])
   {
    var hf_idx=this.horizontal_field_list["record_id"][i];
    var test_number=this.horizontal_field_list["test_number"][i]+"";
    var test_number_idx=this.test_dic["test_number"].indexOf(test_number)
    var hf_name=this.test_dic["test_name"][test_number_idx]
    hf_name=hf_name.split("(")[0];
    var th_=document.createElement("th");this.tr_.appendChild(th_);th_.innerHTML=hf_name;
    var style_="width:"+width_+"px"
    if(i==(n_-1)){style_+=";border-top-right-radius: 10px"}
    th_.setAttribute("style", style_)
   }

   this.tbody_=document.createElement("tbody");this.table_.appendChild(this.tbody_);
   var v_list=this.vertical_field_list["record_id"]
   this.matching_set={"record_id":[], "userid":[], "name":[], "first_name":[], "last_name":[]}

    //alert("acRecordingTests entity_dic 90000-040\n\n"+JSON.stringify(this.entity_dic["first_name"]));

   for(var j in v_list)
   {
     this.matching_set["record_id"].push(v_list[j]);
     var vf_entity_idx=this.vertical_field_list["entity_number"][j]+"";
     var idx_=this.entity_dic["id"].indexOf(vf_entity_idx)
     this.matching_set["userid"].push(this.entity_dic["userid"][idx_]);
     this.matching_set["first_name"].push(this.entity_dic["first_name"][idx_]);
     this.matching_set["last_name"].push(this.entity_dic["last_name"][idx_]);
     this.matching_set["name"].push(this.entity_dic["name"][idx_]);
   }
   //alert("acRecordingTests entity_dic 90000-040\n\n"+JSON.stringify(this.matching_set["name"]));
   this.tbody_.innerHTML=this.create_sorted_structure(v_list, width_, field, container_id, table)

 }

 try{if(this.data != null){this.fill_up_data(this.data, this)}} catch(er){alert(er)}
}


acRecordingTestsCreator.prototype.create_sorted_structure = function(v_list, width_, field, container_id, table)
{
  var s="";
   for(var jj in v_list)
   {
     var j=this.vertical_field_list["record_id"].indexOf(v_list[jj])
     var vf_idx=this.vertical_field_list["record_id"][j];
     var vf_entity_idx=this.vertical_field_list["entity_number"][j]+"";
     var idx_=this.entity_dic["id"].indexOf(vf_entity_idx)
     //var vf_name=this.entity_dic["userid"][idx_]+":"+this.entity_dic["name"][idx_];
     s+="<tr><td>"+this.entity_dic["userid"][idx_]+"</td>"
     s+="<td>"+this.entity_dic["first_name"][idx_]+"</td>";
     s+="<td>"+this.entity_dic["last_name"][idx_]+"</td>";

     for(var i in this.horizontal_field_list["record_id"])
     {
      var hf_idx=this.horizontal_field_list["record_id"][i];//var hf_name=hf_idx;
      var test_number=this.horizontal_field_list["test_number"][i]+"";
      var test_number_idx=this.test_dic["test_number"].indexOf(test_number)
      var hf_type=this.test_dic["test_type"][test_number_idx];
      var field_type_="text"; if(hf_type=="b"){field_type_="checkbox"}
      //alert(hf_type+" "+field_type_)
      s+="<td><input id='tg_"+vf_idx+"_"+hf_idx+"' style='width:"+width_+"px;text-align:right' type_='recording' field='"+field+"' "
      s+=" container_id='"+container_id+"' record_id='new' table='"+table+"'";
      s+="vertical_field='"+this.vertical_field+"' "
      s+="horizontal_field='"+this.horizontal_field+"' type='"+field_type_+"'/></td>"
     }
     s+="<td></tr>";
   }
   return s;
}


acRecordingTestsCreator.prototype.set_data = function(record_id, ll=null)
{
  var dic=this.parent.data;
  //alert("acRecordingTestsCreator 90446-16  "+JSON.stringify(dic));
  var model_=dic["properties"]["table"]
  var container = document.getElementById("content_"+dic["container_id"]);

  var parent_model_=container.my_creator_obj.link_dic["properties"]["table"];
     var dic_={"model":model_, "parent_model":parent_model_, "parent_id": record_id, "number_of_rows": "1000",
               "fields": this.fields,
               "filters":{}, "order_by": {}}
    //alert("90447-150  "+JSON.stringify(dic_));
  this.parent.atm.get_data(call_back_fun=this.fill_up_data, dic_, this)
}

acRecordingTestsCreator.prototype.fill_up_data = function(data, this_obj)
{
  //alert("943016: "+JSON.stringify(data));
  //alert("acRecordingTests this_obj.test_dic 93300-033\n\n"+JSON.stringify(this_obj.test_dic));
  //alert("acRecordingTests this_obj.test_dic 93300-035\n\n"+JSON.stringify(this_obj.horizontal_field_list));
    this_obj.data=data;
    this_obj.members_list={"entity_number":[], "record_id":[]};
    var n_=data[this_obj.fields[0]].length;if(n_<1){return}
    for(var j=0; j<n_; j++)
    {
     try{
         var s_id="tg_"+data["soldiersforevent"][j]+"_"+data["testsforevent"][j];
         var test_number=""+this_obj.horizontal_field_list["test_number"][this_obj.horizontal_field_list["record_id"].indexOf(data["testsforevent"][j])]
         var test_type=this_obj.test_dic["test_type"][this_obj.test_dic["test_number"].indexOf(test_number)]
         var n=1*data["value"][j];
         s_id=getEBI(s_id);s_id.setAttribute("record_id",data["id"][j])
         if(test_type=="b"){if(n==1){s_id.checked=true} else {s_id.checked=false}
         } else{s_id.value=data["value"][j]}
         //alert(s_id.outerHTML)
         } catch(er){}
    }
}


// -- acCom pliance --
function acComplianceCreator(parent){this.parent=parent;this.is_set_unit=true, this.is_set_parameters=false,
                                     this.s_col_time_width=100;this.s_buttons_width=62;this.s_col_width=150;
                                     this.data_dic={}}

acComplianceCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //alert("90358-88 "+JSON.stringify(dic));
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  this.main_div=document.createElement("div");this.main_div.my_creator_obj=this;
  if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="1300"}
  var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"px;"
  this.main_div.setAttribute("style", style_);
  container.appendChild(this.main_div);
  this.main_div.setAttribute("container_id", dic["container_id"]);
  this.main_div.setAttribute("id", dic["properties"]["obj_number"]);
  this.main_div.setAttribute("obj_type", dic["obj_type"]);
  this.weeks=document.createElement("select");
  this.weeks.my_creator=this;
  this.weeks.addEventListener("change", function(event){
    var e=event.target;
    var today_date_d=e.value;
    var today_date_=e.value;
    var y=today_date_.substring(0,4);var m=today_date_.substring(4,6);var d=today_date_.substring(6,8);
    var today_=new Date(y+"-"+m+"-"+d);
    e.my_creator.days.innerHTML="";
    //--days--
    var o = document.createElement("option");o.value="-1";o.innerHTML="-------";e.my_creator.days.appendChild(o);
    for (var j=0;j<7;j++)
    {var o=document.createElement("option");o.value=today_date_;
     today_.setDate(today_.getDate()+1);
     var y=today_.getFullYear()+""; var m=complete_zeros((today_.getMonth()+1)+"",2); var d=complete_zeros(today_.getDate()+"",2);
     var today_date_=y+m+d;
     o.innerHTML=num_to_day(j);e.my_creator.days.appendChild(o);
    }
    e.my_creator.days.value=today_date_d;
    var ec=new Event("change", {bubbles: true});
    e.my_creator.days.dispatchEvent(ec)
  })

  this.main_div.appendChild(this.weeks);
  this.days=document.createElement("select");
  this.days.my_creator=this;
  this.days.addEventListener("change", function(event){
    var e=event.target;
    var units=e.my_creator.units;
    var units_value=units.value;
    if(units_value!=-1 && units_value!=""){var ec=new Event("change", {bubbles: true});units.dispatchEvent(ec)}
    //alert("days change: "+e.value)
  })
  this.main_div.appendChild(this.days);
  this.units=document.createElement("select");
  this.units.my_creator=this;this.units.dic=dic;
  this.main_div.appendChild(this.units);

  this.content_div=document.createElement("div");
  this.main_div.appendChild(this.content_div);
  this.table_=document.createElement("table");

//  border_width_=2;border_color_="blue";
//  style_="border:"+border_width_+"px solid "+border_color_;
//  this.table_.setAttribute("style", style_);

  this.content_div.appendChild(this.table_);
  this.thead_=document.createElement("thead");
  this.table_.appendChild(this.thead_);
  this.tbody_=document.createElement("tbody");
  this.table_.appendChild(this.tbody_);
  this.thead_tr=document.createElement("tr");
  this.thead_.appendChild(this.thead_tr);
  var table_class_=dic["properties"]["table_class"];if(table_class_==null || table_class_==""){table_class_="basic"}
  this.table_.setAttribute("class", table_class_)
  //--
  this.week_conclusion=document.createElement("textarea");
  this.week_conclusion.c=this;
  this.week_conclusion.addEventListener("change", function(event){
   var e=event.target;var c=e.c;c.get_set_data(time_unit=c.data_dic);
  })
  this.day_conclusion=document.createElement("textarea");
  this.day_conclusion.c=this;
  this.day_conclusion.addEventListener("change", function(event){
   var e=event.target;var c=e.c;c.data_dic["conclusion"]=e.value;
   c.get_set_data(time_unit=c.data_dic);
  })
  this.day_conclusion.setAttribute("rows", 3);this.day_conclusion.setAttribute("cols", 114);
  this.week_conclusion.setAttribute("rows", 3);this.week_conclusion.setAttribute("cols", 114);

  var br_=document.createElement("br");
  var week_title=document.createElement("p");week_title.innerHTML="Weekly Summary"
  var day_title=document.createElement("p");day_title.innerHTML="Daily Summary"

  this.main_div.appendChild(day_title);
  this.main_div.appendChild(this.day_conclusion);
  this.main_div.appendChild(week_title);
  this.main_div.appendChild(this.week_conclusion);

  this.units.addEventListener("change", function(event){
    var e=event.target;var c=e.my_creator;c.ll=c.units_structure[e.value];c.llp=e.value;
    //alert("90357-1 "+JSON.stringify(e.my_creator.units_structure));
    //alert("90357 "+e.value+"\n"+JSON.stringify(c.ll));
//    if(c.thead_tr.innerHTML!=""){
//      var id_ = prompt("Are you sure you want to Change Unit? if so type y" , 'No')
//      if(id_ != 'y') {return;}
//    }
    c.thead_tr.innerHTML="";c.tbody_.innerHTML="";
    th=document.createElement("th");c.thead_tr.appendChild(th);th.setAttribute("style",'width:'+this.s_col_time_width+'px');th.innerHTML="Time From"
    th=document.createElement("th");c.thead_tr.appendChild(th);th.setAttribute("style",'width:'+this.s_col_time_width+'px');th.innerHTML="Time To"
    //--
    th=document.createElement("th");c.thead_tr.appendChild(th);th.setAttribute("style",'width:'+this.s_buttons_width+'px');
    var button_add_time=document.createElement("button");button_add_time.c=c;button_add_time.innerHTML="+"
    th.appendChild(button_add_time);
    // creat a new row --
    button_add_time.addEventListener("click", function(event){
        var e=event.target;var c=e.c;
        var row_number_=c.get_next_time_number();
        c.add_new_time_row(row_number=row_number_, t="D")
   })
    //--
    for(var j in c.ll["data"]){th=document.createElement("th");c.thead_tr.appendChild(th);
      th.setAttribute("style",'width:'+c.s_col_width+'px');th.setAttribute("unit_number", j);
      th.innerHTML=c.ll["data"][j]["title"]
    };
    //--
    //c.days.value=c.weeks.value;
    c.get_set_data(time_unit="null");
    //--
  })
  //-- functions --
  for(f in dic["functions"]){var s="this.main_div."+f+"="+dic["functions"][f];eval(s);}
}

acComplianceCreator.prototype.set_weeks_days = function()
{
 if(this.is_set_parameters==true){return}
  var start_date_=""+atm.general.start_date
  var number_of_weeks_=1*atm.general.number_of_weeks
  var y=atm.general.y;var m=atm.general.m;var d=atm.general.d;

  var date_=new Date(y+"-"+m+"-"+d);
  while (date_.getDay()>0){date_.setDate(date_.getDate() -1);}
  var y=date_.getFullYear()+"";var m=complete_zeros((date_.getMonth()+1)+"",2);var d=complete_zeros(date_.getDate()+"",2);
  var s_date=y+m+d;
  // -- weeks --
  var o = document.createElement("option");o.value="-1";o.innerHTML="-------";this.weeks.appendChild(o);
  for (var j=0; j<=number_of_weeks_; j++)
  {var o = document.createElement("option");
       o.value=s_date;
       o.innerHTML="Week "+j+" : "+s_date;
       date_.setDate(date_.getDate()+7)
       var y=date_.getFullYear()+"";
       var m=complete_zeros((date_.getMonth()+1)+"",2);
       var d=complete_zeros(date_.getDate()+"",2);
       var s_date=y+m+d;
       this.weeks.appendChild(o);
   }

  var today_=new Date()
  var y=today_.getFullYear()+"";var m=complete_zeros((today_.getMonth()+1)+"",2);var d=complete_zeros(today_.getDate()+"",2);
  var today_date_d=y+m+d;
  while (today_.getDay()>0){today_.setDate(today_.getDate() -1);}
  var y=today_.getFullYear()+"";var m=complete_zeros((today_.getMonth()+1)+"",2);var d=complete_zeros(today_.getDate()+"",2);
  var today_date_=y+m+d;
  this.weeks.value=today_date_;
  //--days--
   var o = document.createElement("option");o.value="-1";o.innerHTML="-------";this.days.appendChild(o);
   for (var j=0; j<7; j++)
   {var o = document.createElement("option");
    o.value=today_date_;
    today_.setDate(today_.getDate()+1);
    var y=today_.getFullYear()+"";var m=complete_zeros((today_.getMonth()+1)+"",2);var d=complete_zeros(today_.getDate()+"",2);
    var today_date_=y+m+d;
    o.innerHTML=num_to_day(j);this.days.appendChild(o);
   }
   this.days.value=today_date_d;
   var ec=new Event("change", {bubbles: true});
   this.days.dispatchEvent(ec)
}

acComplianceCreator.prototype.add_new_time_row = function(row_number, t="D")
{
  var get_select=function(c, type, tr, j, s_col_width, row_number, onchange_input, colspan=1){
      var td=document.createElement("td");tr.appendChild(td);
      var s_col_width_=s_col_width*colspan
      td.setAttribute("style", 'width:'+s_col_width_+'px')
      td.setAttribute("colspan", colspan)
      var input=document.createElement("select");td.appendChild(input);
      input.setAttribute("unit_number",j);
      input.setAttribute("id",j+"-"+row_number+"-"+t);
      input.onchange=onchange_input;
      input.c=c;
      var o=document.createElement("option");o.value="-1";o.innerHTML="-------";input.appendChild(o);
      if(type=="D")
      {
        input.setAttribute("style",'text-align: right;width:'+s_col_width_+'px')
        var ll=atm.general.platoon_tasks;
        for(var h in ll){var o=document.createElement("option");o.value=h;o.innerHTML=ll[h];input.appendChild(o)};
      } else if(type=="A"){
        input.setAttribute("style",'text-align: right;width:'+(s_col_width_+(5*colspan))+'px')
        var ll=atm.general.company_tasks;
        for(var h in ll){var o=document.createElement("option");o.value=h;o.innerHTML=ll[h];input.appendChild(o)};
      }
  }

  //alert("acComplianceCreator.prototype.add_new_time_row")
  var onchange_input=function(event){
       var e=event.target;
       //alert("90357-101\n"+JSON.stringify(e.c.data_dic));
       var s=e.getAttribute("id");
       var ll=s.split("-");var field_=ll[0];var t=ll[2];var row_number=ll[1];
       //alert(field_ + " " + row_number)
       var t_=e.c.data_dic["times"];
       if(!(row_number in t_)){t_[row_number]={};t_[row_number]["t"]=t};t_[row_number][field_]=e.value;
       //alert("90357-555-2\n"+JSON.stringify(e.c.data_dic));
       e.c.get_set_data(time_unit=e.c.data_dic);
  }

    var tr=document.createElement("tr");this.tbody_.appendChild(tr);
    tr.setAttribute("row_number",row_number);
    // --Time columns --
    var td=document.createElement("td");tr.appendChild(td);td.setAttribute("style",'width:'+this.s_col_time_width+'px')
    var input=document.createElement("input");td.appendChild(input);
    input.setAttribute("style",'width:'+this.s_col_time_width+'px');input.setAttribute("type",'time');
    input.setAttribute("id","time_from-"+row_number+"-"+t);input.onchange=onchange_input;input.c=this;
    //--
    var td=document.createElement("td");tr.appendChild(td);td.setAttribute("style",'width:'+this.s_col_time_width+'px')
    var input=document.createElement("input");td.appendChild(input);
    input.setAttribute("style",'width:'+this.s_col_time_width+'px');input.setAttribute("type",'time');
    input.setAttribute("id","time_to-"+row_number+"-"+t);input.onchange=onchange_input;input.c=this;
    //--
    var td=document.createElement("td");tr.appendChild(td);td.innerHTML+="&nbsp;";
    td.setAttribute("style",'width:'+this.s_buttons_width+'px');
    var button_add_description=document.createElement("button");button_add_description.innerHTML=t;
    button_add_description.c=this;
    td.appendChild(button_add_description);
    button_add_description.addEventListener("click", function(event){
       var e=event.target;var c=e.c;var s=e.innerHTML;
       if(s=="D"){
         for(j in c.ll["data"]){var sid=j+"-"+row_number+"-D";var tr=getEBI(sid).parentNode.parentNode;getEBI(sid).parentNode.outerHTML="";}
         ff_("A", c, tr, c.s_col_width, row_number, onchange_input, get_select)
         //alert(JSON.stringify(c.data_dic["times"][row_number]));
         c.data_dic["times"][row_number]["t"]="A";e.innerHTML="A";
       } else{
         var sid=c.llp+"-"+row_number+"-A";var tr=getEBI(sid).parentNode.parentNode;getEBI(sid).parentNode.outerHTML=""
         for(j in c.ll["data"]){get_select(c, "D", tr, j, c.s_col_width, row_number, onchange_input)}
         c.data_dic["times"][row_number]["t"]="D";e.innerHTML="D";
       }
       c.get_set_data(time_unit=c.data_dic);
    })
    //--
    var button_delete=document.createElement("button");button_delete.innerHTML="--"
    td.appendChild(button_delete);
    button_delete.c=this;
    button_delete.addEventListener("click", function(event){
            var id_ = prompt("Are you sure you want to delete this record? if so type y" , 'No')
            if(id_ != 'y') {return;};
            var e=event.target;var c=e.c;
            var g=e.parentNode.parentNode;g.outerHTML="";
            delete c.data_dic["times"][row_number]
            c.get_set_data(time_unit=c.data_dic);
            // update DB
         })
    var ff_=function(t, c, tr, s_col_width, row_number, onchange_input, get_select){
       if(t=="D"){for(j in c.ll["data"]){get_select(c, t, tr, j, s_col_width, row_number, onchange_input)}
          } else {
          get_select(c, t, tr, c.llp, s_col_width, row_number, onchange_input, colspan=Object.keys(c.ll["data"]).length)
          }
        }
    ff_(t, this, tr, this.s_col_width, row_number, onchange_input, get_select)
}

acComplianceCreator.prototype.get_next_time_number = function()
{
 if(!("last_number" in this.data_dic)){this.data_dic["last_number"]=0}
 var n_=this.data_dic["last_number"]+1;
 this.data_dic["last_number"]=n_;
 this.get_set_data(time_unit=this.data_dic);
 return n_;
}

acComplianceCreator.prototype.get_set_data = function(time_unit)
{
   var time_dim=this.days.value;
   var fun=function(data, ll){
     var obj=ll[0]; var dic_=ll[1]
     //alert("90357-888-1\n "+JSON.stringify(dic_["params"]["time_unit"]));
     //alert("90357-888-2\n "+JSON.stringify(data));
     obj.data_dic=data["time_unit"];
     var cc="";var w="";var kk=data["week_conclusion"];if(kk!=null && kk!=""){cc=kk};obj.week_conclusion.innerHTML=cc;
     if(dic_["params"]["time_unit"]=="null")
     {
      var ts=data["time_unit"]["times"]
      for(var k in ts){
        obj.add_new_time_row(k, ts[k]["t"]);
        for(var j in ts[k]){if(j=="t"){continue};var sid=j+"-"+k+"-"+ts[k]["t"];getEBI(sid).value=ts[k][j];}
       //alert("90357-888-5\n "+k+"\n"+JSON.stringify(ts[k]));
      }
      var c="";var k=data["time_unit"]["conclusion"];if(k!=null && k!=""){c=k};obj.day_conclusion.innerHTML=c;
     }
   }
    var pp_=this.parent.data["properties"];
    var dic=
    {"obj":"TrainingDataProcessing","atm":atm.my_name,"app":atm.my_app,
     "obj_param":{"topic_id":"general","app":atm.my_app},"fun":"compliance_data",
     "params":{"app":atm.my_app,"week_model":pp_["week_table"],"day_model":pp_["day_table"],
               "company_obj_id":atm.company_obj_id,"week_start_day":this.weeks.value, "unit": this.units.value,
               "battalion": atm.general.current_battalion,"time_dim":time_dim,
               "time_unit":time_unit, "week_conclusion":this.week_conclusion.value
             }
     }
    //alert("dic:\n"+JSON.stringify(dic));
    atm.activate_obj_function(fun, dic, [this, dic])
}

acComplianceCreator.prototype.set_units = function()
{
  if(this.is_set_unit==true)
  {
    this.units.innerHTML="";
    var ll=atm.general.get_platoons_ids();ll=ll[3];
    this.units_structure=ll;
    var o=document.createElement("option");o.value="-1";
    o.innerHTML="-------";this.units.appendChild(o);
    for(var j in ll){var o=document.createElement("option");o.value=j;o.innerHTML=ll[j]["title"];this.units.appendChild(o)};
    this.is_set_unit==false;
  }
}

// -- acSearch Table --
function acSearchTableCreator(parent){this.parent=parent;this.is_json_data=false;this.json_record_id=-1;
                                      this.json_last_record_number=0, this.primary_key_list_filter=[],
                                      this.data_table_name=null;
                                      this.parent_model_=null; this.parent_id_=null;
                                      this.parent_model_fk_name_=null;
                                      }

acSearchTableCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //--
  this.is_call_get_data=true;
  var call_get_data=dic["properties"]["call_get_data"];
  if(call_get_data!=null && call_get_data!=""){this.is_call_get_data=call_get_data;}
  //--
  if(!dic["fields"]){dic["fields"]={"1":{"field_title":"1", "field_name":"","field_width":"100","field_align":"left"},
                                    "2":{"field_title":"2", "field_name":"","field_width":"100","field_align":"left"}}}
  if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
  //alert("90356 "+JSON.stringify(dic));
  this.parent.data=dic;
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  this.number_of_rows=dic["properties"]["number_of_rows"];
  this.is_new_button=dic["properties"]["is_new_button"];
  this.is_del_button=dic["properties"]["is_del_button"];
  this.fields=[]
  for(f in dic["fields"]){this.fields.push(dic["fields"][f]["field_name"])}
  //alert("9036-1 "+JSON.stringify(this.fields));
  //--
  this.filter_value="";
  this.search_input_=document.createElement("input");
  this.search_input_.setAttribute("type", "text");
  this.search_input_.setAttribute("id", "search_"+dic["properties"]["obj_number"]);
  var ny_search=parseInt(dic["properties"]["y"])-18
  var nx_search=parseInt(dic["properties"]["x"])+20
  var n_search_width=120;
  if(this.is_new_button=="Yes")
  {
   this.new_button=document.createElement("button");
   this.new_button.setAttribute("id", "new_button_"+dic["properties"]["obj_number"]);
   var nx_nb=nx_search+n_search_width
   this.new_button.setAttribute("style", "position:absolute;left:"+nx_nb+"px;top:"+ny_search+"px;width:40px");
   this.new_button.innerHTML="New";
   this.new_button.my_creator_obj=this;
   container.appendChild(this.new_button);
   this.new_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.is_json_data=this.my_creator_obj.is_json_data;
     container_.my_creator_obj.clear_objects_data();
     container_.setAttribute("parent_id", this.my_creator_obj.json_record_id);
     try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["on_new_record"]);zz.atm=this.atm;zz(event)} catch(er){}
   })
  }
  if(this.is_del_button=="Yes")
  {
   this.del_button=document.createElement("button");
   this.del_button.setAttribute("id", "del_button_"+dic["properties"]["obj_number"]);
   var nx_db=nx_search+n_search_width+40
   this.del_button.setAttribute("style", "position:absolute;left:"+nx_db+"px;top:"+ny_search+"px;width:40px");
   this.del_button.innerHTML="Del";
   this.del_button.my_creator_obj=this;
   container.appendChild(this.del_button);
   this.del_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.delete_record();
     event.target.my_creator_obj.get_data();
     container_.my_creator_obj.clear_objects_data()
   })
  }

  this.search_input_.setAttribute("style", "position:absolute;left:"+nx_search+"px;top:"+ny_search+"px;width:"+n_search_width+"px");
  this.search_input_.addEventListener("keyup", function(event){
     var input=event.target;
     var mco=input.my_creator_obj;
     var filter_field = input.getAttribute("filter_field");
     var filter_value = input.value;
     //alert(filter_field +" : " + filter_value)
     var filter_field_foreign_table = input.getAttribute("filter_field_foreign_table");
     var last_filter_field_ = input.getAttribute("last_filter_field")
     var last_filter_value_ = input.getAttribute("last_filter_value")
     if(mco.is_json_data==true){
           var data_ = [];
           for(var i in mco.tbody.data)
           {if(mco.tbody.data[i][filter_field].toLowerCase().includes(filter_value.toLowerCase())){data_.push(mco.tbody.data[i])}}
           mco.create_tbody(data_, [mco.tbody, mco.parent.data], is_new_data=false)

     }else{
         mco.filter_field=filter_field;
         mco.filter_field_foreign_table=filter_field_foreign_table;
         mco.filter_value=input.value;
         if((last_filter_value_==null) || (last_filter_field_==null) || (last_filter_field_=="") || (last_filter_field_!=filter_field) || (filter_value!=last_filter_value_))
         {
           //alert("904 filter_field= " + filter_field + " input.value= " + input.value + " input.outerHTML= "+input.outerHTML)
           mco.get_data();
           this.setAttribute("last_filter_field", filter_field)
           this.setAttribute("last_filter_value", filter_value)
         }
   }
   })
  container.appendChild(this.search_input_);

  this.table_=document.createElement("table");
  this.table_.setAttribute("class", dic["properties"]["table_class"]);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", dic["properties"]["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);
  //if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
  var style_ = "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"
  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);
  var tr_h=document.createElement("tr");
  tr_h.my_creator_obj=this;
  tr_h.addEventListener("click", function(event){
       var e=event.target;
       var mco=e.parentNode.my_creator_obj;
       var filter_field=e.getAttribute("filter_field");
       var filter_field_foreign_table=e.getAttribute("filter_field_foreign_table");
       mco.search_input_.setAttribute("filter_field", filter_field);
       mco.search_input_.setAttribute("filter_field_foreign_table", filter_field_foreign_table);
       mco.search_input_.setAttribute("placeholder", "Search "+e.innerHTML+"..");
       mco.filter_field=filter_field;
       mco.filter_field_foreign_table=filter_field_foreign_table;
       var field = mco.order_by["field"]
       if(field==filter_field)
       {if(mco.order_by["direction"]=="ascending"){mco.order_by["direction"]="descending"}else{mco.order_by["direction"]="ascending"}}
       else{mco.order_by["field"]=filter_field;mco.order_by["direction"]="ascending"}
       if(mco.is_json_data==true){
         var sorted_data={};var temp_list=[];
         for(var i in mco.tbody.data){temp_list.push(mco.tbody.data[i][mco.order_by["field"]])}
         emp_list = temp_list.sort();if(mco.order_by["direction"]=="ascending"){temp_list=temp_list.reverse()};
         temp_list = [...new Set(temp_list)];
         //  alert("90357 "+JSON.stringify(temp_list));
         var temp_data=JSON.parse(JSON.stringify(mco.tbody.data));var n=1;
         for(var z in temp_list)
         {for(var j in temp_data){if(temp_data[j][mco.order_by["field"]]==temp_list[z]){sorted_data[n]=temp_data[j];n+=1;delete temp_data[j]}}}
         mco.create_tbody(sorted_data, [mco.tbody, this.my_creator_obj.parent.data]);
         var container_id=this.my_creator_obj.parent.data["container_id"];
         var container = document.getElementById("content_"+container_id);
         container.my_creator_obj.clear_objects_data()
       } else{
           e.parentNode.my_creator_obj.get_data();
       }
  })

  tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(tr_h);
  var n__=0;
  this.order_by={"field":"","direction":"ascending"};
  var width__=1*0;
  var nd_=Object.keys(dic["fields"]).length;var n_=0

  for(f in dic["fields"])
  {
    n_+=1;
    var th_=document.createElement("th");
    th_.innerHTML=dic["fields"][f]["field_title"];
    var width_=1*100;try{width_=dic["fields"][f]["field_width"]} catch(er){}
    width__+=1*width_;
    var foreign_table="";if(dic["fields"][f]["foreign_table"]!=null){foreign_table=dic["fields"][f]["foreign_table"]}
    if(n__==0){
      th_.setAttribute("style", "width:"+width_+"px;border-top-left-radius: 15px")
      this.filter_field=dic["fields"][f]["field_name"];
      var f_=dic["fields"][f]["filter"];if(f_!=null&&f_!=""){this.filter_value=f_}
      //alert("dic[fields][f][foreign_table]"+dic["fields"][f]["foreign_table"])
      var x_=dic["fields"][f]["foreign_table"];
      if(x_!=null && x_!=""){this.filter_field_foreign_table=x_;
       var w_=dic["fields"][f]["foreign_table_field"];
       if(w_!=null&&w_!=""){this.filter_field_ft=w_}else{this.filter_field_ft="id"}
      }
      this.search_input_.setAttribute("filter_field", dic["fields"][f]["field_name"]);
      this.search_input_.setAttribute("filter_field_foreign_table", foreign_table);
      this.search_input_.setAttribute("placeholder", "Search "+dic["fields"][f]["field_title"]+"..");
      this.order_by["field"]=dic["fields"][f]["field_name"];
    }
    else{if(n_==nd_){width_=width_*1+1*17} th_.setAttribute("style", "width:"+width_+"px;")}; n__+=1;
    tr_h.appendChild(th_);
    th_.setAttribute("filter_field", dic["fields"][f]["field_name"])
    th_.setAttribute("filter_field_foreign_table", foreign_table)
  }
  width__+=1*17; style_ += width__+"px"
  th_.setAttribute("style", "width:"+width_+"px;border-top-right-radius: 15px");
  this.tbody=document.createElement("tbody");
  this.tbody.setAttribute("id", "tbody_"+dic["properties"]["obj_number"]);
  this.tbody.setAttribute("style", "display: block;height: "+dic["properties"]["height"]+"px;overflow-y: auto;overflow-x: hidden;");

  this.tbody.my_creator_obj=this;
  this.table_.appendChild(this.tbody);
  this.table_.my_creator_obj=this;
  //alert(style_)
  this.table_.setAttribute("style", style_);
  this.search_input_.my_creator_obj=this;

  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in dic["properties"]){this.table_.setAttribute(s, dic["properties"][s])}
   else{this.table_.setAttribute(s, "")}}
  container.appendChild(this.table_);
  this.table_.onclick=this.row_click;
  for(f in dic["functions"]){if(f!="onclick" && f!="on_new_record"){var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}
}

acSearchTableCreator.prototype.set_primary_key_list_filter = function(list=[])
{
 this.primary_key_list_filter=list;
}

acSearchTableCreator.prototype.get_data = function(data_table_name=null,parent_model=null, parent_id=null,
                                                   company_obj_id=null, parent_model_fk_name=null)
{
  //alert("acSearchTableCreator.prototype.get_data = "+ 1)
  //alert("parent_model_fk_name\n"+parent_model_fk_name)
  //alert("9090-12 data_table_name\n"+data_table_name)
  var dic=this.parent.data;
  //alert("10000  "+JSON.stringify(dic));
  try{var f_=dic["functions"]["before_get_data"]; var s_="var zz="+f_;eval(s_); zz(dic);} catch(er){}
  // alert("0000  "+JSON.stringify(dic["fields"]));
  var container = document.getElementById("content_"+dic["container_id"]);
  if(this.data_table_name!=null && data_table_name!=null && data_table_name!=""){this.data_table_name=data_table_name}
  else {this.data_table_name=container.getAttribute("table")}
  try{
    var parent_id_="";
    var model_=container.my_creator_obj.link_dic["properties"]["table"];
    try{
        if(this.parent_model_==null){
           if(parent_model!=null){this.parent_model_=parent_model}
           else {this.parent_model_=container.my_creator_obj.link_dic["properties"]["parent_table"]}
        }
        if(this.parent_model_==null){this.parent_model_="";}

    } catch(er){alert(er)};
    // var record_id_=container.getAttribute("record_id");

    var c_parent_id=null;
    try{var c_parent_id=container.getAttribute("parent_id")}catch(er){}
    if(parent_id!=null){this.parent_id_=parent_id}
    else if(c_parent_id!=null && c_parent_id!="new" && c_parent_id!="-1"){this.parent_id_=c_parent_id}
    if(this.parent_id==null){this.parent_id_=""}
//alert(153)
    // if(this.parent_id_=="" || this.parent_id_==null){return}
//alert(18)

    var dic_={"model":this.data_table_name, "number_of_rows":this.number_of_rows, "fields":this.fields}
  } catch(er){alert("er90132: "+er)}

  var fun=function(data, ttbody_){
    try{var f_=dic["functions"]["on_get_data"]; var s_="var hh="+f_;eval(s_); hh(data, dic__);} catch(er){}
    console.log("get_data search table inside call back function")
    // alert("9081 data: "+JSON.stringify(data));

    try{if(Object.keys(data).length==0){return;}
    var n_=data[dic_["fields"][0]].length;
    } catch (er) {}
    //alert(dic_["fields"][0])
    //alert(JSON.stringify(data[dic_["fields"][0]]))

    console.log("get_data search table inside call back function n=" + n_)
    var s='';
    for(i=0; i<n_; i++)
    {
     s+='<tr id_="'+data[data["pkf_name"]][i]+'" row="'+i+'"'+'>';
     for(j in dic_["fields"])
     {
       var f=dic_["fields"][j];
       var width_=200;var field_align_="left";
       for(z in dic["fields"]){
        if(dic["fields"][z]["field_name"]==f){width_=dic["fields"][z]["field_width"];field_align_=dic["fields"][z]["field_align"]}
        }
        s+='<td style="width:'+width_+'px;text-align:'+field_align_+'">'+data[f][i]+'</td>'
     };
     s+='</tr>';
    }
    //alert("9008 get_data search table inside call back function s=" + s)
    ttbody_.innerHTML=s;

    //alert(ttbody_.my_creator_obj)
    //alert(ttbody_.my_creator_obj.is_json_data)
    //ttbody_.my_creator_obj.is_json_data=false;
  }
  var dic__=[]; for(i in dic_["fields"]){dic__.push(dic_["fields"][i])}

  var container_id=this.parent.data["container_id"];
  var container_dic=this.parent.tab.tab_objects[container_id];
  //alert("9065 container_dic= \n"+JSON.stringify(container_dic))
  this.multiple_select_fields=[]
  for(o in container_dic)
  {
     if(container_dic[o]["obj_type"]=="acObj")
      {
        var f=container_dic[o]["properties"]["field"];
        if(f!=null){
            if(container_dic[o]["obj_name"]=="acInput" || container_dic[o]["obj_name"]=="acTextarea" || container_dic[o]["obj_name"]=="acRichText")
            {if(!dic__.includes(f)){dic__.push(f)}
            } else if (container_dic[o]["obj_name"]=="acSelect"){var ps_=container_dic[o]["properties"];
             if("multiple" in ps_){this.multiple_select_fields.push(f)}else{if(!dic__.includes(f)){dic__.push(f)}}
            }
        }
      }
      else if(container_dic[o]["obj_name"]=="acUploadFile") {
        //alert("90601-23 \n"+JSON.stringify(container_dic[o]))
        var f=container_dic[o]["properties"]["field"];
        if(f!=null){if(!dic__.includes(f)){dic__.push(f)}}
      }
  }

  var dic__={"model":this.data_table_name, "parent_model": this.parent_model_, "parent_id":this.parent_id_,
             "number_of_rows":this.number_of_rows, "multiple_select_fields": this.multiple_select_fields,
             "filters":{}, "order_by":this.order_by, "fields":dic__
             }

//  alert("90449-99-25: "+JSON.stringify(dic__)+"\n");

   dic__["filters"][this.filter_field]={"value":this.filter_value, "foreign_table":this.filter_field_foreign_table}

//    var x_=dic["fields"][f]["foreign_table"];
//      if(x_!=null && x_!=""){this.filter_field_foreign_table=x_;
//       var w_=dic["fields"][f]["foreign_table_field"];
//       if(w_!=null&&w_!=""){this.filter_field_ft=w_}else{this.filter_field_ft="id"}
//      }

// this.filter_field
// this.filter_value
// this.filter_field_ft
// this.filter_field_foreign_table

  //alert("90449-99-26: "+JSON.stringify(dic__)+"\n");

   // I do not have to put the next line it but good readability
   var x_=this.filter_field_foreign_table;var w_=this.filter_field_ft;
   if(x_!=null && x_!="" && (w_!=null && w_!=""))
   {dic__["filters"][this.filter_field]["field"]=w_} else{dic__["filters"][this.filter_field]["field"]="id"}

  //alert("90449-99-27: "+JSON.stringify(dic__)+"\n");

  //alert("90447-77 dic__ \n"+JSON.stringify(dic__["filters"]))
  console.log("get_data search table ", JSON.stringify(dic__))

    for(j in dic["fields"])
    {
       // alert("90447-77-1 dic["+j+"] \n"+JSON.stringify(dic["fields"][j]));
       if("filter" in dic["fields"][j] && dic["fields"][j]["filter"]!="")
       {
          var fn_=dic["fields"][j]["field_name"]
          if(!(fn_ in dic__["filters"]) || dic__["filters"][fn_]["value"]=="")
          {
            var foreign_table_="";
            if ("foreign_table" in dic["fields"][j])
            {
              foreign_table_=dic["fields"][j]["foreign_table"];
            }
            dic__["filters"][fn_]={"value":dic["fields"][j]["filter"], "field":"id", "foreign_table": foreign_table_}
          }
       }
    }

  if(this.company_obj_id_==null){
     if(company_obj_id!=null){this.company_obj_id_=company_obj_id}
  }
  dic__["company_obj_id"]=this.company_obj_id_;

  dic__["primary_key_list_filter"]=this.primary_key_list_filter
        if(this.parent_model_fk_name_==null){
            if(parent_model_fk_name!=null){this.parent_model_fk_name_=parent_model_fk_name}
        }
        if(this.parent_model_fk_name_==null){this.parent_model_fk_name_=""}
  dic__["parent_model_fk_name"]=this.parent_model_fk_name_;

//  alert("90447-777 dic__ \n"+JSON.stringify(dic__))

  this.parent.atm.get_data(call_back_fun=fun, dic__, this.tbody)
}

acSearchTableCreator.prototype.get_data_json = function(data_table_name=null,record_id=null,company_obj_id=null)
{
  //alert("9087:\ndata_table_name:"+data_table_name+"\nrecord_id: "+record_id)
  var dic=this.parent.data;
  //alert("9088:\ndic:"+JSON.stringify(dic["properties"]["obj_number"]))
  var container = document.getElementById("content_"+dic["container_id"]);
  container.my_creator_obj.link_dic["properties"]["table"]=data_table_name;
  container.setAttribute("json_manager_obj_number", dic["properties"]["obj_number"])
  this.json_record_id=record_id;
  var dic__ = {"model_name":data_table_name, "record_id": record_id}
  //alert("input "+JSON.stringify(dic__));
  this.parent.atm.get_data_json(call_back_fun=this.create_tbody, dic__, [this.tbody, dic])
}

acSearchTableCreator.prototype.get_next_number = function()
{this.json_last_record_number=1*this.json_last_record_number+1; return this.json_last_record_number;}

acSearchTableCreator.prototype.update_json_record = function(dic)
{
   var container = getEBI("content_"+dic["container_id"])
   //alert('9084 dic_data= '+ JSON.stringify(dic));
   if(dic["record_id"]=="new"){dic["record_id"]=this.get_next_number();this.tbody.data[dic["record_id"]]={};
    container.setAttribute("record_id", dic["record_id"])
   }
   //alert('9086 dic_data= '+ JSON.stringify(this.tbody.data));
   //alert('90861 field_old_value= '+ JSON.stringify(this.tbody.data[dic["record_id"]][dic["field"]]));
   this.tbody.data[dic["record_id"]][dic["field"]]=dic["field_value"];

   //alert('9085 dic_data= '+ JSON.stringify(this.tbody.data));

   var dic_={"model":dic["model"],"parent_model":"","pkey":dic["parent_id"],"parent_pkey":"-1","type":""}
   dic_["fields"]={};dic_["fields"][dic["model"]]=this.tbody.data;
   // alert('9087 dic_data= '+ JSON.stringify(dic_));

   this.parent.tab.parent.save_data(container, dic_, is_json_data=true)
   this.create_tbody(this.tbody.data,[this.tbody, this.parent.data])
   //alert('9087 dic_data= '+ JSON.stringify(this.tbody.data[dic["record_id"]]));
}

acSearchTableCreator.prototype.create_tbody = function(data, ll, is_new_data=true)
{
    var tbody=ll[0];if(is_new_data){tbody.data=data;var max=0;
      for(var j in data){if(j>max){max=j}};tbody.my_creator_obj.json_last_record_number=max;
    };
    var dic=ll[1];
    //alert("recieved data  "+JSON.stringify(data));
    var s='';
    for(var i in data)
    {
     s+='<tr id_="'+i+'" row="'+i+'"'+'>';
     for(var z in dic["fields"])
     {
      var k=dic["fields"][z]["field_name"];
      var width=50; var align="left";width=dic["fields"][z]["field_width"];align=dic["fields"][z]["field_align"]
      var v=""; if(data[i][k]!=null){v=data[i][k]}
      s+='<td style="width:'+width+'px;text-align:'+align+'">'+v+'</td>'
     }
     s+='</tr>';
    }
    tbody.innerHTML=s;
    tbody.my_creator_obj.is_json_data=true;
}

acSearchTableCreator.prototype.row_click = function(event)
{
 var e=event.target;
 while(e.tagName!="TR"){e=e.parentNode;}
 //alert(this.my_creator_obj.table_.outerHTML)

 var n_row=e.getAttribute("row"); if(n_row==null){return;}
 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);
 //alert("1:\n"+container.outerHTML)
 //alert(container_id)
 var result={}
 var dic=this.my_creator_obj.tbody.data;
 if(this.my_creator_obj.is_json_data==true){
   container.my_creator_obj.clear_objects_data()
   var id_=e.getAttribute("id_")
   result=JSON.parse(JSON.stringify(dic[id_]))
   result["id"]=id_;
   container.my_creator_obj.set_objects_data(result, is_json_data=true);
 } else{
   for(var f in dic){
    if(f != "pkf_name"){result[f]=dic[f][n_row]} else {result["pkf_name"]=dic[f]}
   };
   //alert(JSON.stringify(result))

   //alert(container.my_creator_obj.set_objects_data)

   container.my_creator_obj.set_objects_data(result, is_json_data=false);
 }
 container.my_creator_obj.current_record_data=result;
 container.my_creator_obj.link_content.setAttribute("parent_id", this.my_creator_obj.json_record_id);
 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event);} catch(er){}
 var p=e.parentNode;for (let i=0;i<p.children.length;i++){p.children[i].style.backgroundColor=""};
 e.style.backgroundColor="lightblue"
}

acSearchTableCreator.prototype.editor_properties_func = function(tab, tab_properties_)
{
 //alert(tab.active_obj)
 var field_setting=tab.parent.buttons[tab.active_obj.data["parent_obj_name"]]["sub_buttons"][tab.active_obj.data["element_name"]]["field_setting"]
 //alert(field_setting)
 var fields=tab.active_obj.data["fields"];
 //alert(JSON.stringify(fields));
 var nav=document.createElement("div");
 var add_del_btns=document.createElement("div");
 this.add_btn=document.createElement("button");this.del_btn=document.createElement("button");
 this.add_btn.innerHTML="Add"; this.add_btn.tab=tab;
 this.del_btn.innerHTML="Delete"; this.del_btn.tab=tab; this.del_btn.nav=nav;
 this.add_btn.onclick=function (event){
   var n_=Object.keys(tab.active_obj.data["fields"]).length+1
   tab.active_obj.data["fields"][n_]={"field_name":"","field_title":"","field_width":"100","field_align":"left"};
   tab.parent.save();
 }
 this.del_btn.onclick=function (event){
   //alert(this.nav.active_property_num)
   delete tab.active_obj.data["fields"][this.nav.active_property_num];
   tab.parent.save();
 }
 add_del_btns.appendChild(add_btn);add_del_btns.appendChild(del_btn);
 nav.setAttribute("style","cursor:pointer");
 var nav_detail=document.createElement("div");
 nav.setAttribute("class", "scrollmenu");
 for(f in fields){var a=document.createElement("a");a.innerHTML=f;nav.appendChild(a);}
 nav.onclick=function(event)
 {
  var e=event.target;
  nav_detail.innerHTML="";
  f=e.innerHTML
  this.active_property_num=f
  //alert(JSON.stringify(this.fields[f]));
  var table=document.createElement("table");
  var tr=document.createElement("tr");table.appendChild(tr);
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="property"
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="value"
  for(p_ in field_setting) // this.fields[f])
  {
   var p=field_setting[p_]
   var v=this.fields[f];var tr=document.createElement("tr");table.appendChild(tr);
   var td=document.createElement("td");tr.appendChild(td);td.innerHTML=p
   var td=document.createElement("td");tr.appendChild(td);var input=document.createElement("input");
   input.setAttribute("size","10");td.appendChild(input);
   input.setAttribute("field",f);input.setAttribute("property",p);
   try{if(this.fields[f][p]!=null){input.value=this.fields[f][p]}} catch(er){}
  }
  nav_detail.appendChild(table);
 }
 nav.fields=fields;
 nav_detail.onchange=function(event){
   event.preventDefault();
   var e=event.target;
   var f=e.getAttribute("field")
   var p=e.getAttribute("property")
   //alert(e.outerHTML);
   //alert(e.value);
   tab.active_obj.data["fields"][f][p]=e.value;
   tab.parent.save();
 }
 nav_detail.tab=tab;
 tab_properties_.appendChild(document.createElement("br"));
 tab_properties_.appendChild(add_del_btns);
 tab_properties_.appendChild(nav);
 tab_properties_.appendChild(nav_detail);
}


// -- acMSearchTableCreator --
function acMSearchTableCreator(parent){
  this.parent=parent;
  // alert(JSON.stringify(this.parent.data));
}

acMSearchTableCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  if(!dic["fields"]){dic["fields"]={"1":{"field_title":"1", "field_name":"","field_width":"100","field_align":"left"},
                                    "2":{"field_title":"2", "field_name":"","field_width":"100","field_align":"left"}}}
  if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
  //alert(JSON.stringify(dic));
  this.parent.data=dic;
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  this.number_of_rows=dic["properties"]["number_of_rows"];
  this.is_new_button=dic["properties"]["is_new_button"];
  this.is_del_button=dic["properties"]["is_del_button"];
  this.fields=[]
  for(f in dic["fields"]){this.fields.push(dic["fields"][f]["field_name"])}
  //--
  this.table_=document.createElement("table");
  this.table_.setAttribute("class", dic["properties"]["table_class"]);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", dic["properties"]["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);

  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in dic["properties"]){this.table_.setAttribute(s, dic["properties"][s])}
   else{this.table_.setAttribute(s, "")}}
  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);
  var tr_h=document.createElement("tr");tr_h.my_creator_obj=this;
  tr_h.addEventListener("click", function(event){
     var e=event.target;
     var filter_field=e.getAttribute("filter_field")
     var mco = e.parentNode.my_creator_obj
     var field = mco.order_by["field"]
     if(field==filter_field)
     {if(mco.order_by["direction"]=="ascending"){mco.order_by["direction"]="descending"}else{mco.order_by["direction"]="ascending"}}
     else{mco.order_by["field"]=filter_field;mco.order_by["direction"]="ascending"}
     e.parentNode.my_creator_obj.get_data();
  })
  tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(tr_h);
  //--
  var ny_search=parseInt(dic["properties"]["y"])-18
  var nx_search=parseInt(dic["properties"]["x"])+4
  var table_width=0;
  this.filters={};this.order_by={"field":"","direction":"ascending"};
  var nf=0
  for(f in dic["fields"])
  {
    nx_search+=1*1
    var width_=100;try{width_=dic["fields"][f]["field_width"]} catch(er){}
    var sr=document.createElement("input");sr.my_creator_obj=this;
    container.appendChild(sr);
    sr.setAttribute("type", "text");
    sr.setAttribute("id", "search_"+f+"_"+dic["properties"]["obj_number"]);
    sr.setAttribute("style", "position:absolute;left:"+nx_search+"px;top:"+ny_search+"px;width:"+width_+"px");
    var foreign_table="";if(dic["fields"][f]["foreign_table"]!=null){foreign_table=dic["fields"][f]["foreign_table"]}
      sr.setAttribute("filter_field", dic["fields"][f]["field_name"]);
      sr.setAttribute("filter_field_foreign_table", foreign_table);
    this.filters[dic["fields"][f]["field_name"]]={"value":"","foreign_table":foreign_table}
    if(nf==0){this.order_by["field"]=dic["fields"][f]["field_name"]};nf+=1;
      sr.setAttribute("placeholder", "Search "+dic["fields"][f]["field_title"]+"..");
      nx_search+=1*width_
    sr.addEventListener("keyup", function(event){
         var input=event.target;var mco=input.my_creator_obj;
         var filter_field = input.getAttribute("filter_field");
         mco.filters[filter_field]["value"]=input.value;
         mco.get_data();
    })
    var th_=document.createElement("th");
    th_.innerHTML=dic["fields"][f]["field_title"];

    table_width+=1*width_;
    var style_="width:"+width_+"px;"
    if(f==1){style_+=";border-top-left-radius: 10px"} else if (f==Object.keys(dic["fields"]).length) {{style_+=";border-top-right-radius: 10px"}}
    th_.setAttribute("style", style_);
    tr_h.appendChild(th_);
    th_.setAttribute("filter_field", dic["fields"][f]["field_name"])
    th_.setAttribute("filter_field_foreign_table", foreign_table)
  }

  this.tbody=document.createElement("tbody");
  this.tbody.setAttribute("id", "tbody_"+dic["properties"]["obj_number"]);
  this.tbody.setAttribute("style", "display: block;height: "+dic["properties"]["height"]+"px;overflow-y: auto;overflow-x: hidden;");
  this.table_.appendChild(this.tbody);
  this.table_.my_creator_obj=this;
  this.table_.setAttribute("style", "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+table_width+"px");

  if(this.is_new_button=="Yes")
  {
   this.new_button=document.createElement("button");
   this.new_button.setAttribute("id", "new_button_"+dic["properties"]["obj_number"]);
   var nx_nb=nx_search
   nx_search+=40
   this.new_button.setAttribute("style", "position:absolute;left:"+nx_nb+"px;top:"+ny_search+"px;width:40px");
   this.new_button.innerHTML="New";
   this.new_button.my_creator_obj=this;
   container.appendChild(this.new_button);
   this.new_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.clear_objects_data()
   })
  }
  if(this.is_del_button=="Yes")
  {
   this.del_button=document.createElement("button");
   this.del_button.setAttribute("id", "del_button_"+dic["properties"]["obj_number"]);
   var nx_db=nx_search
   this.del_button.setAttribute("style", "position:absolute;left:"+nx_db+"px;top:"+ny_search+"px;width:40px");
   this.del_button.innerHTML="Del";
   this.del_button.my_creator_obj=this;
   container.appendChild(this.del_button);
   this.del_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.delete_record();
     event.target.my_creator_obj.get_data();
     container_.my_creator_obj.clear_objects_data()
   })
  }

  container.appendChild(this.table_);
  this.table_.onclick=this.row_click;
  for(f in dic["functions"]){if(f!="onclick"){var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}
}

acMSearchTableCreator.prototype.get_data = function(data_table_name=null,parent_model=null,parent_id=null)
{
  var dic=this.parent.data;
  //alert(JSON.stringify(dic));
  var container = document.getElementById("content_"+dic["container_id"]);
  //alert(container.outerHTML);
  if(data_table_name!==null){this.data_table_name=data_table_name} else {this.data_table_name=container.getAttribute("table")}
  try{
    var parent_id_="";
    //alert(this.data_table_name)
    var model_=container.my_creator_obj.link_dic["properties"]["table"];
    //alert(model_)
    try{
        if(parent_model!=null){parent_model_=parent_model} else {
          var parent_model_=container.my_creator_obj.link_dic["properties"]["parent_table"]
          }
          if(parent_model_==null){var parent_model_="";}
        } catch(er){};
    // var record_id_=container.getAttribute("record_id");
    if(parent_id!=null){parent_id_=parent_id} else {parent_id_=container.getAttribute("parent_id")}
    if(parent_id_=="" || parent_id_==null){return}
    var dic_={"model":this.data_table_name, "number_of_rows":this.number_of_rows, "fields":this.fields}
  } catch(er){alert("er9012: "+er)}

  var fun=function(data, ttbody_){

  //alert("get_data search table inside call back function")

    //alert(JSON.stringify(data));
    //alert(dic_["fields"][0])
    //alert(JSON.stringify(data[dic_["fields"][0]]))
    var n_=data[dic_["fields"][0]].length;
    //alert(n_)
  //alert("get_data search table inside call back function n=" + n_)
    var s='';
    for(i=0; i<n_; i++)
    {
     s+='<tr id_="'+data["id"][i]+'" row="'+i+'"'+'>';
     for(j in dic_["fields"])
     {
       var f=dic_["fields"][j];
       var width_=200;var field_align_="left";
       for(z in dic["fields"]){
        if(dic["fields"][z]["field_name"]==f){width_=dic["fields"][z]["field_width"];field_align_=dic["fields"][z]["field_align"]}
        }
        s+='<td style="width:'+width_+'px;text-align:'+field_align_+'">'+data[f][i]+'</td>'
     };
     s+='</tr>';
    }

  //alert("get_data search table inside call back function s=" + s)

    ttbody_.innerHTML=s;
  }

  //alert(JSON.stringify(dic_));
  var container_id=this.parent.data["container_id"];
  //alert(container_id);

  var container_dic=this.parent.tab.tab_objects[container_id];
  var dic__=[]; for(i in dic_["fields"]){dic__.push(dic_["fields"][i])}
  for(o in container_dic)
  {if(container_dic[o]["obj_type"]=="acObj" && container_dic[o]["obj_name"]=="acInput"){
      var f=container_dic[o]["properties"]["field"];if(!dic__.includes(f)){dic__.push(f)}}
  }
  //alert(JSON.stringify(dic__));
  var dic__={"model":this.data_table_name, "parent_model": parent_model_, "parent_id":parent_id_,
             "number_of_rows":this.number_of_rows,
             "filters": this.filters,"order_by":this.order_by,
             "fields":dic__}

  try{var f_=dic["functions"]["before_get_data"]; var s_="var zz="+f_;eval(s_); zz(dic__);} catch(er){}

  //alert("AAAAAAAAAAAA\n"+JSON.stringify(dic__));
  console.log("get_data search table ", JSON.stringify(dic__))
  this.parent.atm.get_data(call_back_fun=fun, dic__, this.tbody)
}

acMSearchTableCreator.prototype.row_click = function(event)
{
 var e=event.target;
 while(e.tagName!="TR"){e=e.parentNode;}
 //alert(e.outerHTML)
 //alert(this.my_creator_obj.table_.outerHTML)
 var n_row=e.getAttribute("row")
 if(n_row==null){return;}
 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);
 //alert(container_id)
 var result={}
 //alert(JSON.stringify(this.my_creator_obj.tbody.data))
 var dic=this.my_creator_obj.tbody.data;
 for(f in dic){result[f]=dic[f][n_row]}

 container.my_creator_obj.set_objects_data(result);
 container.my_creator_obj.current_record_data=result;

 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event)} catch(er){}
 var p=e.parentNode;for(let i=0;i<p.children.length;i++){p.children[i].style.backgroundColor=""};
 e.style.backgroundColor="lightblue";
}

acMSearchTableCreator.prototype.editor_properties_func = function(tab, tab_properties_)
{
 //alert(tab.active_obj)
 var field_setting=tab.parent.buttons[tab.active_obj.data["parent_obj_name"]]["sub_buttons"][tab.active_obj.data["element_name"]]["field_setting"]
 //alert(field_setting)
 var fields=tab.active_obj.data["fields"];
 //alert(JSON.stringify(fields));
 var nav=document.createElement("div");
 var add_del_btns=document.createElement("div");
 this.add_btn=document.createElement("button");this.del_btn=document.createElement("button");
 this.add_btn.innerHTML="Add"; this.add_btn.tab=tab;
 this.del_btn.innerHTML="Delete"; this.del_btn.tab=tab; this.del_btn.nav=nav;
 this.add_btn.onclick=function (event){
   var n_=Object.keys(tab.active_obj.data["fields"]).length+1
   tab.active_obj.data["fields"][n_]={"field_name":"","field_title":"","field_width":"100","field_align":"left"};
   tab.parent.save();
 }
 this.del_btn.onclick=function (event){
   //alert(this.nav.active_property_num)
   delete tab.active_obj.data["fields"][this.nav.active_property_num];
   tab.parent.save();
 }
 add_del_btns.appendChild(add_btn);add_del_btns.appendChild(del_btn);
 nav.setAttribute("style","cursor:pointer");
 var nav_detail=document.createElement("div");
 nav.setAttribute("class", "scrollmenu");
 for(f in fields){var a=document.createElement("a");a.innerHTML=f;nav.appendChild(a);}
 nav.onclick=function(event)
 {
  var e=event.target;
  nav_detail.innerHTML="";
  f=e.innerHTML
  this.active_property_num=f
  //alert(JSON.stringify(this.fields[f]));
  var table=document.createElement("table");
  var tr=document.createElement("tr");table.appendChild(tr);
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="property"
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="value"
  for(p_ in field_setting) // this.fields[f])
  {
   var p=field_setting[p_]
   var v=this.fields[f];var tr=document.createElement("tr");table.appendChild(tr);
   var td=document.createElement("td");tr.appendChild(td);td.innerHTML=p
   var td=document.createElement("td");tr.appendChild(td);var input=document.createElement("input");
   input.setAttribute("size","10");td.appendChild(input);
   input.setAttribute("field",f);input.setAttribute("property",p);
   try{if(this.fields[f][p]!=null){input.value=this.fields[f][p]}} catch(er){}
  }
  nav_detail.appendChild(table);
 }
 nav.fields=fields;
 nav_detail.onchange=function(event){
   event.preventDefault();
   var e=event.target;
   var f=e.getAttribute("field")
   var p=e.getAttribute("property")
   //alert(e.outerHTML);
   //alert(e.value);
   tab.active_obj.data["fields"][f][p]=e.value;
   tab.parent.save();
 }
 nav_detail.tab=tab;
 tab_properties_.appendChild(document.createElement("br"));
 tab_properties_.appendChild(add_del_btns);
 tab_properties_.appendChild(nav);
 tab_properties_.appendChild(nav_detail);
}


// -- acGeneralLedgerCreator --
function acGeneralLedgerCreator(parent){this.parent=parent;}  // alert(JSON.stringify(this.parent.data));

acGeneralLedgerCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  //alert(JSON.stringify(dic));

  if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
  //alert(JSON.stringify(dic));
  this.parent.data=dic;
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  this.number_of_rows=dic["properties"]["number_of_rows"];
  this.model=dic["properties"]["table"];
  this.is_new_button=dic["properties"]["is_new_button"];
  this.is_del_button=dic["properties"]["is_del_button"];
  this.table_=document.createElement("table");
  this.table_.addEventListener("change", function(event){
    var e=event.target
    e.setAttribute("model", this.getAttribute("model"))
    e.setAttribute("content_type", "accounting")
   })

  this.table_.setAttribute("class", dic["properties"]["table_class"]);
  this.table_.setAttribute("model", this.model);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", dic["properties"]["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);
  this.table_.setAttribute("style", "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:400px");
  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);
  var tr_h=document.createElement("tr");
  tr_h.my_creator_obj=this;
  tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(tr_h);
    var th_an=document.createElement("th");th_an.innerHTML="Acc #";
    th_an.setAttribute("style","width:50px;border-top-left-radius: 15px");tr_h.appendChild(th_an);
    var th_a=document.createElement("th");th_a.innerHTML="Account name";
    th_a.setAttribute("style","width:250px;");tr_h.appendChild(th_a);
    var th_m=document.createElement("th");th_m.innerHTML="Amount";
    th_m.setAttribute("style","width:100px;border-top-right-radius: 15px");
    tr_h.appendChild(th_m);
  this.tbody=document.createElement("tbody");
  this.tbody.setAttribute("id", "tbody_"+dic["properties"]["obj_number"]);
  this.tbody.setAttribute("style", "display: block;");
  this.table_.appendChild(this.tbody);
  this.table_.my_creator_obj=this;

  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in dic["properties"]){this.table_.setAttribute(s, dic["properties"][s])}
   else{this.table_.setAttribute(s, "")}}

  container.appendChild(this.table_);
  this.table_.onclick=this.row_click;
  for(f in dic["functions"]){if(f!="onclick"){var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}
}

acGeneralLedgerCreator.prototype.set_accounts = function(dic=null,obj_nums=[])
{
 if(dic!=null){this.accounts_dic=dic}
 else{
  if(obj_nums.length>1)
  {
   for(j in obj_nums){var id__=obj_nums[j]+""; getEBI(id__).my_creator_obj.set_accounts();};
   return;
  }
 }

 this.tbody.innerHTML=""
 for(a in this.accounts_dic)
 {
  var tr=document.createElement("tr");this.tbody.appendChild(tr);
  var tdan=document.createElement("td");tdan.setAttribute("style","width:50px");tdan.innerHTML=a;tr.appendChild(tdan);
  var tda=document.createElement("td");tda.setAttribute("style","width:250px");tda.innerHTML=this.accounts_dic[a];tr.appendChild(tda);
  var tdi=document.createElement("td");tdi.setAttribute("style","width:100px");
  var i=document.createElement("input");
  var table_id=this.parent.data["properties"]["obj_number"]
  i.setAttribute("id", table_id+"_"+a);
  i.setAttribute("container_id", this.table_.getAttribute("container_id"));
  i.setAttribute("style","width:100px;text-align:right");
  i.setAttribute("type_","Input");
  i.setAttribute("field","amount");
  i.setAttribute("account",a);
  tdi.appendChild(i);tr.appendChild(tdi);
 }

}

acGeneralLedgerCreator.prototype.set_data = function(parent_id, obj_nums=[])
{
 //alert(parent_id)
 //alert(JSON.stringify(obj_nums));
  var fun=function(data, obj_nums){
    try{
        if(obj_nums.constructor === Array){
         for(j in obj_nums)
         {
            var table_id=obj_nums[j]+"";
            var n_=data["account"].length;
            var s='';
            for(i=0; i<n_; i++)
            {
             try{
               var i_=document.getElementById(table_id+"_"+data["account"][i])
               i_.setAttribute("record_id", data["id"][i])
               i_.value=data["amount"][i]
               } catch(er) {}
            }
         }
        }
    }catch(er){}
  }
  this.set_accounts(null,obj_nums);
  var model_ = document.getElementById("content_"+this.parent.data["container_id"]).getAttribute("table");
  var dic_={"model":this.model, "number_of_rows":1000, "fields":["account", "amount", "comment"],
             "parent_model":model_ , "parent_id": parent_id, filters:{}, order_by:{}}

  if(obj_nums.length<1){obj_nums.push(this.parent.data["properties"]["obj_number"])}
  this.parent.atm.get_data(call_back_fun=fun, dic_, obj_nums)
}

acGeneralLedgerCreator.prototype.row_click = function(event)
{
 var e=event.target;
 while(e.tagName!="TR"){e=e.parentNode;}
 //alert("1: "+e.outerHTML)
 //alert(this.my_creator_obj.table_.outerHTML)
 var n_row=e.getAttribute("row")
 if(n_row==null){return;}

 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);
 //alert(container_id)

 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event)} catch(er)
 {//alert("er9013: "+er)
 }

}


// -- acDETableCreator --
function acDETableCreator(parent){
  this.parent=parent;
  // alert(JSON.stringify(this.parent.data));
}

acDETableCreator.prototype.create_obj = function()
{
  var dic=this.parent.data;
  if(!dic["fields"]){dic["fields"]={"1":{"field_title":"1", "field_name":"","field_width":"100","field_align":"left","field_type":"text"},
                                    "2":{"field_title":"2", "field_name":"","field_width":"100","field_align":"left","field_type":"checkbox"}}}
  if(!(dic["properties"]["table_class"])){dic["properties"]["table_class"]="basic"}
  //alert(JSON.stringify(dic));
  this.parent.data=dic;
  this.filters={};this.order_by={"field":"","direction":"ascending"};
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  this.number_of_rows=dic["properties"]["number_of_rows"];
  this.is_new_button=dic["properties"]["is_new_button"];
  this.is_del_button=dic["properties"]["is_del_button"];
  this.model=dic["properties"]["table"];
  this.fields=[]
  for(f in dic["fields"]){this.fields.push(dic["fields"][f]["field_name"])}
  //--
  this.table_=document.createElement("table");
  this.table_.my_creator_obj=this;
  this.table_.setAttribute("class", dic["properties"]["table_class"]);
  this.table_.setAttribute("container_id", dic["container_id"]);
  this.table_.setAttribute("id", dic["properties"]["obj_number"]);
  this.table_.setAttribute("obj_type", dic["obj_type"]);
  this.table_.setAttribute("type", dic["element_name"]);
  //--
  this.table_.setAttribute("model", this.model);
  this.table_.addEventListener("change", function(event){
    var e=event.target;
    e.setAttribute("model", this.getAttribute("model"))
    e.setAttribute("record_id",e.parentNode.parentNode.getAttribute("id_"));
    // alert("de: "+e.outerHTML)
   })
  //--
  for (i in dic["attributes"]){var s=dic["attributes"][i];
   if(s in dic["properties"]){this.table_.setAttribute(s, dic["properties"][s])}
   else{this.table_.setAttribute(s, "")}}

  this.thead=document.createElement("thead");
  this.thead.setAttribute("style", "display: block;overflow-x: hidden;");
  this.table_.appendChild(this.thead);
  var tr_h=document.createElement("tr");tr_h.my_creator_obj=this;
  tr_h.setAttribute("style","cursor:pointer");
  this.thead.appendChild(tr_h)
  this.tbody=document.createElement("tbody");
  this.tbody.setAttribute("id", "tbody_"+dic["properties"]["obj_number"]);
  this.tbody.setAttribute("style", "display: block;height: "+dic["properties"]["height"]+"px;overflow-y: auto;overflow-x: hidden;");
  this.table_.appendChild(this.tbody);

  var table_width=0;
  this.tr_body_=document.createElement("tr");this.tbody.appendChild(this.tr_body_);this.tr_body_.setAttribute("id_","");
  this.tr_body_.setAttribute("id",dic["properties"]["obj_number"]+"_1");
  for(f in dic["fields"])
  {
    var width_=100;try{width_=dic["fields"][f]["field_width"]} catch(er){}
    var width_i=width_-6
    var th_=document.createElement("th");
    th_.innerHTML=dic["fields"][f]["field_title"];
    table_width+=1*width_;
    th_.setAttribute("style", "width:"+width_+"px;");
    tr_h.appendChild(th_);

    var td=document.createElement("td");
    td.setAttribute("style","width:"+width_+"px");
    this.tr_body_.appendChild(td);

    if(dic["fields"][f]["field_type"]=="select")
    {var i=document.createElement("select")} else{var i=document.createElement("input")}

    //alert(JSON.stringify(dic["fields"][f]));

    i.setAttribute("id", dic["fields"][f]["field"]);
    i.setAttribute("field", dic["fields"][f]["field"]);
    i.setAttribute("container_id", this.table_.getAttribute("container_id"));

    i.setAttribute("style","width:"+width_i+"px;text-align:"+dic["fields"][f]["field_align"]);
    i.setAttribute("type",dic["fields"][f]["field_type"]);
    td.appendChild(i);
  }
  //--
  var ny_search=parseInt(dic["properties"]["y"])-18
  var nx_search=parseInt(dic["properties"]["x"])+4
  this.table_.setAttribute("style", "position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+table_width+"px");
  if(this.is_new_button=="Yes")
  {
   this.new_button=document.createElement("button");
   this.new_button.setAttribute("id", "new_button_"+dic["properties"]["obj_number"]);
   var nx_nb=nx_search
   nx_search+=40
   this.new_button.setAttribute("style", "position:absolute;left:"+nx_nb+"px;top:"+ny_search+"px;width:40px");
   this.new_button.innerHTML="New";
   this.new_button.my_creator_obj=this;
   container.appendChild(this.new_button);
   this.new_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.clear_objects_data()
   })
  }
  if(this.is_del_button=="Yes")
  {
   this.del_button=document.createElement("button");
   this.del_button.setAttribute("id", "del_button_"+dic["properties"]["obj_number"]);
   var nx_db=nx_search
   this.del_button.setAttribute("style", "position:absolute;left:"+nx_db+"px;top:"+ny_search+"px;width:40px");
   this.del_button.innerHTML="Del";
   this.del_button.my_creator_obj=this;
   container.appendChild(this.del_button);
   this.del_button.addEventListener("click", function(event){
     var container_=event.target.parentNode;
     container_.my_creator_obj.delete_record();
     event.target.my_creator_obj.get_data();
     container_.my_creator_obj.clear_objects_data()
   })
  }
  container.appendChild(this.table_);
  this.table_.onclick=this.row_click;
  for(f in dic["functions"]){if(f!="onclick"){var s="this.table_."+f+"="+dic["functions"][f];eval(s);}}
//----
//---
}

acDETableCreator.prototype.set_data = function(parent_id, obj_nums=[])
{
 //alert(parent_id)
 //alert(JSON.stringify(obj_nums));
  var fun=function(data, obj_nums){
    try{
        if(obj_nums.constructor === Array){
         for(j in obj_nums)
         {
            var table_id=obj_nums[j]+"";
            var n_=data["account"].length;
            var s='';
            for(i=0; i<n_; i++)
            {
             try{
               var i_=document.getElementById(table_id+"_"+data["account"][i])
               i_.setAttribute("record_id", data["id"][i])
               i_.value=data["amount"][i]
               } catch(er) {}
            }
         }
        }
    }catch(er){}
  }
  this.set_accounts(null,obj_nums);
  var model_ = document.getElementById("content_"+this.parent.data["container_id"]).getAttribute("table");
  var dic_={"model":this.model, "number_of_rows":1000, "fields":["account", "amount", "comment"],
             "parent_model":model_ , "parent_id": parent_id, filters:{}, order_by:{}}

  if(obj_nums.length<1){obj_nums.push(this.parent.data["properties"]["obj_number"])}
  this.parent.atm.get_data(call_back_fun=fun, dic_, obj_nums)
}

acDETableCreator.prototype.row_click = function(event)
{
 var e=event.target;
 while(e.tagName!="TR"){e=e.parentNode;}
 //alert("1: "+e.outerHTML)
 //alert(this.my_creator_obj.table_.outerHTML)
 var n_row=e.getAttribute("row")
 if(n_row==null){return;}

 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);
 //alert(container_id)

 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event)} catch(er)
 {//alert("er9013: "+er)
 }

}

acDETableCreator.prototype.editor_properties_func = function(tab, tab_properties_)
{
 //alert(tab.active_obj)
 var field_setting=tab.parent.buttons[tab.active_obj.data["parent_obj_name"]]["sub_buttons"][tab.active_obj.data["element_name"]]["field_setting"]
 //alert(JSON.stringify(field_setting));
 var fields=tab.active_obj.data["fields"];
 // alert(JSON.stringify(fields));
 var nav=document.createElement("div");
 var add_del_btns=document.createElement("div");
 this.add_btn=document.createElement("button");this.del_btn=document.createElement("button");
 this.add_btn.innerHTML="Add"; this.add_btn.tab=tab;
 this.del_btn.innerHTML="Delete"; this.del_btn.tab=tab; this.del_btn.nav=nav;
 this.add_btn.onclick=function (event){
   var n_=Object.keys(tab.active_obj.data["fields"]).length+1
   tab.active_obj.data["fields"][n_]={"field_name":"","field_title":"","field_width":"100","field_align":"left"};
   tab.parent.save();
 }
 this.del_btn.onclick=function (event){
   //alert(this.nav.active_property_num)
   delete tab.active_obj.data["fields"][this.nav.active_property_num];
   tab.parent.save();
 }
 add_del_btns.appendChild(add_btn);add_del_btns.appendChild(del_btn);
 nav.setAttribute("style","cursor:pointer");
 var nav_detail=document.createElement("div");
 nav.setAttribute("class", "scrollmenu");
 for(f in fields){var a=document.createElement("a");a.innerHTML=f;nav.appendChild(a);}
 nav.onclick=function(event)
 {
  var e=event.target;
  nav_detail.innerHTML="";
  f=e.innerHTML
  this.active_property_num=f
  //alert(f);
  //alert(JSON.stringify(this.fields[f]));

  var table=document.createElement("table");
  var tr=document.createElement("tr");table.appendChild(tr);
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="property"
  var th=document.createElement("th");tr.appendChild(th);th.innerHTML="value"

  //alert(JSON.stringify(field_setting));

  for(var p_ in field_setting) // this.fields[f])
  {
   var p=field_setting[p_]
   var v=this.fields[f][p_];
   //alert(p_ + " - " + p+ " = " + v);
   var tr=document.createElement("tr");table.appendChild(tr);
   var td=document.createElement("td");tr.appendChild(td);td.innerHTML=p_
   var td=document.createElement("td");tr.appendChild(td);
   if(p.length==0)
   {
     var input=document.createElement("input");
     input.setAttribute("size","10");
   } else
   {
     var input=document.createElement("select");
     for (j in p){var o = document.createElement("option");o.value=p[j];o.innerHTML=p[j];input.appendChild(o);}
     input.setAttribute("style", "width:100px;");
   }
     td.appendChild(input);
     input.setAttribute("field",f);
     input.setAttribute("property",p_);
     try{if(this.fields[f][p_]!=null){input.value=this.fields[f][p_]}} catch(er){}
     // alert(input.outerHTML)
  }
  nav_detail.appendChild(table);
 }

 nav.fields=fields;
 nav_detail.onchange=function(event){
   event.preventDefault();
   var e=event.target;
   //alert(e.outerHTML);
   var f=e.getAttribute("field")
   var p=e.getAttribute("property")
   //alert(f+" : "+p+" : "+e.value);
   //alert(JSON.stringify(tab.active_obj.data["fields"][f]));
   tab.active_obj.data["fields"][f][p]=e.value;
   //alert(JSON.stringify(tab.active_obj.data["fields"][f]));
   tab.parent.save();
 }
 nav_detail.tab=tab;
 tab_properties_.appendChild(document.createElement("br"));
 tab_properties_.appendChild(add_del_btns);
 tab_properties_.appendChild(nav);
 tab_properties_.appendChild(nav_detail);
}

// -- acChart Creator --
function acChartCreator(parent){this.parent=parent;}

acChartCreator.prototype.create_obj = function()
{
  //alert("I am a creator of acPlugin Chart")
  //--
  var dic=this.parent.data;
  //alert("9079 dic:"+JSON.stringify(dic));
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
 this.chart=document.createElement("div");
 this.chart.setAttribute("container_id", dic["container_id"]);
 this.chart.setAttribute("id", dic["properties"]["obj_number"]);
 this.chart.setAttribute("obj_type", dic["obj_type"]);
 this.chart.setAttribute("type", dic["element_name"]);
 if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
 if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
 var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"px;"
 style_+="height:"+height_+"px";
 this.chart.setAttribute("style", style_);

 this.chart.my_creator_obj=this;

 container.appendChild(this.chart);

   var chart_type={"type":"pies", "title":"abcde", "x":{"data":[]},"series":{
                    "y1":{"data":[], "color":[219, 64, 82], "name":"import"}}}
  //alert(JSON.stringify(chart_type));

   for(var i=0; i<3;i++)
   {
     chart_type["x"]["data"].push(i)
     chart_type["series"]["y1"]["data"].push(i+Math.random())
   }
  //alert("90711 chart_type:"+JSON.stringify(chart_type));
 this.set_chart_data(chart_type);
}

acChartCreator.prototype.set_chart_data = function(chart_type)
{
  var pp_=this.parent.data["properties"];
//  alert("90213-550\n"+JSON.stringify(chart_type));

  var data_=[];
  chart_attributes={"pies":{"type_name":"type", "type_value":"pie", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "areas":{"type_name":"type", "type_value":"scatter", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "scatter":{"type_name":"mode", "type_value":"markers", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "scatter3d":{"type_name":"mode", "type_value":"markers", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "mscatter":{"type_name":"mode", "type_value":"markers", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "lines":{"type_name":"mode", "type_value":"lines", "marker_attribute":"size", "marker_attribute_value":"8"},
                    "bars":{"type_name":"type", "type_value":"bar", "marker_attribute":"opacity", "marker_attribute_value":"0.7"},
                    "histogram":{"type_name":"type", "type_value":"histogram"},
                    "bubbles":{"type_name":"", "type_value":""},
                    "indicator":{"type_name":"mode", "type_value":"indicator"}
                   }

//    alert("90223-550 chart_attributes\n"+JSON.stringify(chart_attributes));
//    chart_type["type_name"]=chart_attributes[chart_type["type"]]["type_name"];
//    chart_type["type_name_value"]=chart_attributes[chart_type["type"]]["type_value"]

    data={"type":chart_type["type"],
         "type_name":chart_attributes[chart_type["type"]]["type_name"],
         "type_name_value":chart_attributes[chart_type["type"]]["type_value"],
         "title":chart_type["title"],
         "x":{"data":chart_type["x"]},
         "series": chart_type["series"]
        }

//   alert("90712 data:"+JSON.stringify(data));

 if(chart_type["type"]=='pies')
 {
  var n_=0;
  section=1/(Object.keys(data["series"]).length+1.5)
  var k_=section*1.25;
  var annotations_=[]
  for(y in data["series"]){
    var trace = {}
    trace["values"]=data["series"][y]["data"];
    trace["labels"]=data["x"]["data"];
    trace["txt"]=data["series"][y]["name"];
    trace["textposition"]='inside';
    trace["domain"]={column: n_};n_+=1;
    trace["name"]=data["series"][y]["name"];
    trace["hoverinfo"]='lable';
    trace["hole"]=.4;
    trace[data["type_name"]]=data["type_name_value"];
    data_.push(trace)
    annotations_.push({font: {size: 20}, showarrow: false, text: data["series"][y]["name"], x: k_, y: 0.5});k_+=section;
  };
  var layout = {"title": data["title"],
                "annotations": annotations_,
                "height": 400, "width": 600, "showlegend": false,
                "grid": {rows: 1, columns: n_}
               }
   Plotly.newPlot(this.chart, data_, layout);
 } else if (chart_type["type"]=='areas') {
   for(y in data["series"]){
    var trace = {}
    trace["x"]=data["x"]["data"];
    trace["y"]=data["series"][y]["data"];
    trace[data["type_name"]]=data["type_name_value"];
    trace["fill"]='tozeroy';
    trace["mode"]='none';
    data_.push(trace);
   };
   Plotly.newPlot(this.chart, data_, layout );
 }
 else if(chart_type["type"]=='scatter') {
    var trace = {}
    trace["x"]=chart_type["x"];
    trace["y"]=chart_type["y"];
    trace[data["type_name"]]=data["type_name_value"];
    trace["type"]="scatter";
    trace["marker"] = { size: 5 }
    data_.push(trace);
   var layout = {title: data["title"],
                 xaxis: {title: chart_type["x-axis-title"], range: chart_type["x-axis-range"]},
                 yaxis: {title: chart_type["y-axis-title"], range: chart_type["y-axis-range"]}}

   //alert(JSON.stringify(layout));
   Plotly.newPlot(this.chart, data_, layout );
 }
 else if(chart_type["type"]=='scatter3d') {

   //alert("90711 chart_type"+JSON.stringify(chart_type));
   //alert("90712 data:"+JSON.stringify(data));
 var data_ = [];
 for(var k in chart_type["c"])
 {
  //alert(k+":\n"+JSON.stringify(chart_type["c"][k]));
  var trace = {mode: 'markers',name: "Group "+k,
	marker: {size: 3,symbol: 'circle',line: {color: 'rgba(217, 217, 217, '+(k*0.14)+')',width: k*0.5},opacity: 0.8},
	type: 'scatter3d'
  };

  for(var h in chart_type["c"][k])
  { //alert(k+":"+h+":\n"+JSON.stringify(chart_type["c"][k][h]));
   trace[h] = chart_type["c"][k][h]
  }
  data_.push(trace)
 }

  // alert(JSON.stringify(layout));
 var layout = {margin: {l:0,r: 0,b: 0,t: 0}};
 Plotly.newPlot(this.chart, data_, layout );
 }
 else if(chart_type["type"]=='mscatter') {
   //alert("90213-550-1\n"+JSON.stringify(chart_type["data"]));
   var d = chart_type["data"];
   var data_ = [];
   for(var i in d){
    //alert(i + "\n" + JSON.stringify(d[i])+"\n"+d[i]["x"]+"\n"+d[i]["y"])
     var t_ = {x: d[i]["x"], y: d[i]["y"], name: i+"", type: 'scatter' };
     t_[data["type_name"]]=data["type_name_value"];
     t_["marker"] = { size: 3 }
     //alert(JSON.stringify(t_));
     data_.push(clone_dic(t_))
   }
   //alert(JSON.stringify(data_));
   var layout = {title: data["title"],
                 xaxis: {title: chart_type["x-axis-title"], range: chart_type["x-axis-range"]},
                 yaxis: {title: chart_type["y-axis-title"], range: chart_type["y-axis-range"]}}
   Plotly.newPlot(this.chart, data_, layout );
 }
 else if(chart_type["type"]=='histogram') {
    var trace = {}
    trace["x"]=chart_type["x"]["data"];
    trace["type"]="histogram";
    trace["xbins"]={end:chart_type["xbins"]["end"],size:chart_type["xbins"]["size"],start:chart_type["xbins"]["start"]}
    data_.push(trace);
    var layout = {title: data["title"],
                  xaxis: {title: chart_type["x-axis-title"]},
                  yaxis: {title: chart_type["y-axis-title"]}}
   //alert(JSON.stringify(data_));
   //modeBarButtonsToRemove: ['toImage'], displaylogo: false,

  var ss=pp_["remove_buttons"];if(ss==null || ss=="false"){ss=true}else{ss=false}

  Plotly.newPlot(this.chart, data_, layout, {displayModeBar: ss});
 }
 else if(chart_type["type"]=='bubbles'){
    var trace1 = {
      x: chart_type["x"],
      y: chart_type["series"]["y1"]["data"],
      mode: 'markers',
      marker: {
        color: ['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
        size: chart_type["size"]
      }
    };
    var data = [trace1];
    var layout = { title: 'Marker Size', showlegend: false, height: 600,  width: 600,
      xaxis: {title: "x", range: [0, 50], autorange: false},
      yaxis: {title: "y", range: [0, 50], autorange: false}
    };

    var interval_ = this.parent.data["properties"]["interval"]
    if (interval_==null || interval_=="")
    {Plotly.newPlot(this.chart, data, layout);}
    else{
        var obj_number=this.parent.data["properties"]["obj_number"]
        eval("zzz_"+obj_number+"="+chart_type["timer_function"])
        //alert("zzz_"+obj_number)
        try{
           this.parent.atm.set_fun_for_timer(
              fun_name="zzz_"+obj_number,
              fun_ref="zzz_"+obj_number+"(obj)",
              interval=1*this.parent.data["properties"]["interval"],
              obj=[this.chart, data, layout]
           )
        }catch(er){alert(er)}
    }
 }
 else if(chart_type["type"]=='lines'){
     //alert("90233-550-1\n"+JSON.stringify(chart_type));
    var data = [];

    var ss=pp_["remove_markers"];if(ss==null || ss=="true"){ss=""}else{ss=false}
    for(var k in chart_type["series"])
    {
         var trace = {
          x: chart_type["x"],
          y: chart_type["series"][k]["data"],
          name: chart_type["series"][k]["name"],
          mode: "lines+"+ss,
          type: "scatter"
        };
        data.push(trace);
    }

    var ss=pp_["remove_hovermode"];if(ss==null || ss=="true"){ss=false}else{ss=true}
    var layout = {
      title: chart_type["title"],
      hovermode: ss,
      xaxis: {title: chart_type["x-axis-title"], range: chart_type["x-axis-range"]},
      yaxis: {title: chart_type["y-axis-title"], range: chart_type["y-axis-range"]}

//, range: chart_type["x-axis-range"]
            //"range": [/* Start range */, /* End range */]




    };

    var ss=pp_["remove_buttons"];if(ss==null || ss=="false"){ss=true}else{ss=false}
    Plotly.newPlot(this.chart, data, layout, {displayModeBar: ss});
 }
 else if(chart_type["type"]=='bars'){
    //alert("90243-550-1\n"+JSON.stringify(chart_type));
    var data = [];
    for(var k in chart_type["series"])
    {
         var trace = {
          x: chart_type["x"]["data"],
          y: chart_type["series"][k]["data"],
          name: chart_type["series"][k]["name"],
          type: "bar"
        };
        data.push(trace);
    }
    var layout = {
      title: chart_type["title"],
      barmode: 'relative'
    };
    Plotly.newPlot(this.chart, data, layout);
 }
 else if(chart_type["type"]=='indicator') {
    var width_ = chart_type["width"]; if(width_==null || width_==""){width_=400}
    var height_ = chart_type["height"]; if(height_==null || height_==""){height_=400}
    var rows_ = chart_type["rows"]; if(rows_==null || rows_==""){rows_=1}
    var columns_ = chart_type["columns"]; if(columns_==null || columns_==""){columns_=400}
//    shape: "angular" or "bullet"

   //alert(JSON.stringify(chart_type));
   var ctc = chart_type["charts"];
   for(var i in ctc)
   {
        if(ctc[i]["shape"]=="bullet"){var domain_={ x: ctc[i]["x"], y: ctc[i]["y"]}
        } else {var domain_={ row: ctc[i]["row"], column: ctc[i]["column"]}}
        var trace = {
          title:ctc[i]["title"],
          type:"indicator",
          delta: { reference: ctc[i]["reference"] },
          value: ctc[i]["value"],
          mode: ctc[i]["mode"],
          gauge: {shape: ctc[i]["shape"],axis: { visible: ctc[i]["visible"], range: ctc[i]["range"]}},
          domain: domain_
        }
        data_.push(trace);
    }

   //alert(JSON.stringify(data_));
    var layout = {width: width_, height: height_,
                margin: { t: 25, b: 25, l: 25, r: 25 },
                grid: { rows: rows_, columns: columns_, pattern: "independent" },
                template: {data: {indicator: [{}]}}
                };
   Plotly.newPlot(this.chart, data_, layout);
 }
 else {
  //alert(chart_type["type"])
   for(y in data["series"]){
     var trace = {}
     trace["x"]=data["x"]["data"];
     trace["y"]=data["series"][y]["data"];
     trace[data["type_name"]]=data["type_name_value"];
     trace["marker"]={"color":"rgb("+data["series"][y]["color"][0]+", "+data["series"][y]["color"][1]+", "+data["series"][y]["color"][2]+")"}
     trace["marker"][data["marker_attribute"]]=data["marker_attribute_value"]
     trace["name"]=data["series"][y]["name"]
     trace["line"]={color: 'rgb('+data["series"][y]["color"][0]+', '+data["series"][y]["color"][1]+', '+data["series"][y]["color"][2]+')', width: 2}
     data_.push(trace)
   }
    var layout = {title: data["title"],
                xaxis: {title: chart_type["x-axis-title"],
//                type: "date",
                 range: chart_type["x-axis-range"], autorange: true
//                 range: ['2022-10-07 09:30:00-04:00', '2022-10-12 16:00:00-04:00'], autorange: true
                },
                yaxis: {title: chart_type["y-axis-title"], range: chart_type["y-axis-range"], autorange: true,
                       showgrid: true, zeroline: true, showline: true, showticklabels: true}
               }
 }
}

acChartCreator.prototype.get_data = function()
{
  //alert("get Data")
}

// -- acCandleCreator --
function acCandleCreator(parent){this.parent=parent;}

acCandleCreator.prototype.create_obj = function()
{
  //alert("I am a creator of acPlugin Candle")
  //--
  var dic=this.parent.data;
  //alert(JSON.stringify(dic));
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
  this.candle=document.createElement("canvas");
  this.div_candle=document.createElement("div");
  this.div_candle.appendChild(this.candle);
  this.ctx = this.candle.getContext('2d');
  var width_="400";if("width" in dic["properties"]){width_=dic["properties"]["width"]}
  var height_="400";if("height" in dic["properties"]){height_=dic["properties"]["height"]}

  //this.ctx.canvas.width = width_; this.ctx.canvas.height = height_;
  this.div_candle.setAttribute("container_id", dic["container_id"]);
  this.div_candle.setAttribute("id", dic["properties"]["obj_number"]);
  this.div_candle.setAttribute("obj_type", dic["obj_type"]);
  this.div_candle.setAttribute("type", dic["element_name"]);
  var border_width_="1";if("border_width" in dic["properties"]){var border_width_=dic["properties"]["border_width"]}
  var border_color_="#000000;";if("border_color" in dic["properties"]){var border_color_=dic["properties"]["border_color"]}
  var style_="width:"+width_+"px;height:"+height_+"px;border:"+border_width_+"px solid "+border_color_;
  this.candle.setAttribute("style", style_);

  var s_div="position:absolute;width:"+width_+"px;height:"+height_+"px;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;"
  this.div_candle.setAttribute("style", s_div);

  this.div_candle.my_creator_obj=this;
  container.appendChild(this.div_candle);
  //alert(this.ctx.canvas.width + " : " + this.ctx.canvas.height)

  this.chart = new Chart(this.ctx, {type: 'candlestick',
				                    data: {datasets: [{label: 'CHRT - Chart.js Corporation', data: []}]}
			                        });

  this.div_candle.onclick=this.candle_click;
  for(f in dic["functions"]){if(f!="onclick"){var s="this.div_candle."+f+"="+dic["functions"][f];eval(s);}}
}

acCandleCreator.prototype.set_data = function(params={})
{
  //alert(JSON.stringify(params));
  var bar_count = params["bar_count"]; if(bar_count==null || bar_count==""){bar_count=100}
  var bar_data = params["bar_data"];if(bar_data==null || bar_data==""){bar_data=null}
  var line_data = params["line_data"];if(line_data==null || line_data==""){line_data=null}
  var chart_title = params["chart_title"]; if(chart_title==null || chart_title==""){chart_title="Chart"}
  var chart_title_2 = params["chart_title_2"]; if(chart_title_2==null || chart_title_2==""){chart_title_2="Chart"}

			var getRandomInt = function(max) {
				return Math.floor(Math.random() * Math.floor(max));
			};

			function randomNumber(min, max) {
				return Math.random() * (max - min) + min;
			}

			function randomBar(date, lastClose) {
			    var factor = 0.02
				var open = +randomNumber(lastClose * (1-factor), lastClose * (1+factor)).toFixed(2);
				var close = +randomNumber(open * (1-factor), open * (1+factor)).toFixed(2);
				var high = +randomNumber(Math.max(open, close), Math.max(open, close) * 1.05).toFixed(2);
				var low = +randomNumber(Math.min(open, close) * 0.95, Math.min(open, close)).toFixed(2);
				return {
					x: date.valueOf(),
					o: open,
					h: high,
					l: low,
					c: close
				};
			}

			function getRandomData(dateStr, count) {
				//var date = luxon.DateTime.fromRFC2822(dateStr);
				var date = luxon.DateTime.local(2022,10,1,9,30,0);
				var data = [randomBar(date, 130)];

				while (data.length < count) {
					date = date.plus({minutes: 1});
				//alert(date.weekday+" : "+date.day+" : "+date.hour+" : "+date.minute)
					if (date.weekday <= 5) {
				//alert(date.weekday+" : "+date.day+" : "+date.hour+" : "+date.minute)
					   if((date.hour>=10 && date.hour<=15) || (date.hour==9 && date.minute>=30) || (date.hour==16 && date.minute==0))
					   {
						   data.push(randomBar(date, data[data.length - 1].c));
				//alert(date.weekday+" : "+date.day+" : "+date.hour+" : "+date.minute)
						   if(date.hour==16 && date.minute==0){
						    date = date.plus({hours: 17}); date = date.plus({minutes: 29});
						   }
					   }
					}
				}
				//alert(JSON.stringify(data));
				return data;
			}

  if(bar_data == null){var bar_data = getRandomData(this.initialDateStr, bar_count);}

  var dic=this.parent.data;
  var scale_type="linear";if("scale_type" in dic["properties"]){scale_type=dic["properties"]["scale_type"]}

  this.chart.config.options.scales.y.type = scale_type;
  this.chart.config.data.datasets = [{label: chart_title,data: bar_data}]
  if(line_data != null){
    // alert(JSON.stringify(line_data));
  this.chart.config.data.datasets=[{label: chart_title,data: bar_data},{label: chart_title_2,type: 'line',data: line_data}]}
    // alert(JSON.stringify(this.chart.config.data.datasets));
    //alert(JSON.stringify(bar_data));
  this.chart.update();
}

acCandleCreator.prototype.candle_click = function(event)
{
 var e=event.target;
 var container_id=this.my_creator_obj.parent.data["container_id"];
 var container = document.getElementById("content_"+container_id);

 try{eval('var zz='+this.my_creator_obj.parent.data["functions"]["onclick"]);zz(event);} catch(er){}
}


// -- acHeatmapCreator --
function acHeatmapCreator(parent){this.parent=parent;}

acHeatmapCreator.prototype.create_obj = function()
{
  //alert("I am a creator of acPlugin Chart")
  //--
  var dic=this.parent.data;
  //alert(JSON.stringify(dic));
  //--
  var container = document.getElementById("content_"+dic["container_id"]);
  //--
 this.heatmap=document.createElement("div");
 this.heatmap.setAttribute("container_id", dic["container_id"]);
 this.heatmap.setAttribute("id", dic["properties"]["obj_number"]);
 this.heatmap.setAttribute("obj_type", dic["obj_type"]);
 this.heatmap.setAttribute("type", dic["element_name"]);
 if("width" in dic["properties"]){var width_=dic["properties"]["width"]} else {var width_="400"}
 if("height" in dic["properties"]){var height_=dic["properties"]["height"]} else {var height_="400"}
 var style_="position:absolute;left:"+dic["properties"]["x"]+"px;top:"+dic["properties"]["y"]+"px;width:"+width_+"px;"
 style_+="height:"+height_+"px";
 this.heatmap.setAttribute("style", style_);
 this.heatmap.my_creator_obj=this;
 container.appendChild(this.heatmap);

   var chart_type={"x": ['A', 'B', 'C', 'D', 'E'], "y": ['W', 'X', 'Y', 'Z'], "z": [
      [0.10, 0.20, 0.3, 0.4, 0.50],
      [0.60, 0.70, 0.75, 0.75, 1.00],
      [0.75, 0.75, 0.75, 0.75, 0.75],
      [0.80, 0.90, 0.00, 0.75, 0.00]
    ]}
  //alert(JSON.stringify(chart_type));
 this.set_chart_data(chart_type);
}

acHeatmapCreator.prototype.set_chart_data = function(chart_type)
{
    var xValues = chart_type["x"];
    var yValues = chart_type["y"];
    var zValues = chart_type["z"];

    var colorscaleValue = [[0, '#3D9970'],[1, '#001f3f']];

    var data = [{
        x: xValues, y: yValues, z: zValues,
        type: 'heatmap', colorscale: colorscaleValue, showscale: false
    }];

    var layout = {
      title: this.parent.data["properties"]["title"],annotations: [],
      xaxis: {ticks: '', side: 'top'},
      yaxis: {ticks: '', ticksuffix: ' ', width: 700, height: 700, autosize: false}
    };

    for ( var i = 0; i < yValues.length; i++ ) {
      for ( var j = 0; j < xValues.length; j++ ) {
        var currentValue = zValues[i][j];
        if (currentValue != 0.0) { var textColor = 'white';}else{var textColor = 'black'; }
        var result = {
          xref: 'x1', yref: 'y1',
          x: xValues[j], y: yValues[i], text: zValues[i][j],
          font: {family: 'Arial', size: 12, color: 'rgb(50, 171, 96)'},
          showarrow: false,
          font: { color: textColor }
        };
        layout.annotations.push(result);
      }
    }

    Plotly.newPlot(this.heatmap, data, layout );
    try{eval("call_back="+this.parent.data["functions"]["onclick"]);} catch(e){alert(e)}
    try{
         this.heatmap.on('plotly_click', function(data){
             let point = data.points[0];
             let data_=data["points"][0]["data"]
             call_back(point, data_)
        });
    } catch(er) {alert(er)}
}

acHeatmapCreator.prototype.get_data = function()
{
  //alert("get Data")
}

// -- TabContent --
function TabContent(tab, container, link_dic, is_on_click=true, is_link=null){
 //alert("9090-22\n"+JSON.stringify(link_dic));
 //alert("9090-21\n"+container.outerHTML)
 var pp_=link_dic["properties"]
 this.is_link=is_link;
 this.tab=tab;
 this.link_dic=link_dic;
 this.link_number=pp_["link_number"];
 this.content_type=pp_["content_type"];
 this.is_json_data=false;
 //-
 this.link_content=document.createElement("div");
 this.link_content.my_creator_obj=this
 this.link_content.setAttribute("id", "content_"+this.link_number);
 this.link_content.setAttribute("table", pp_["table"]);
 this.link_content.setAttribute("record_id", "new");
 this.link_content.setAttribute("parent_id", "new");
 this.link_content.setAttribute("link_number", this.link_number);
 this.link_content.setAttribute("type", "container");
 this.link_content.setAttribute("class", "tabcontent");
 this.link_content.tab=tab;
 this.link_content.tab_content_id=this.link_number;
 if(is_link){var width=100} else {var width=pp_["width"];}
 if("height" in pp_){var height_=pp_["height"]} else {var height_="1000"}
 //alert("JSON.stringify(pp_)\n\n"+ JSON.stringify(pp_));
 var background_color_="white";
 var color_="black";
 if(pp_["background-color"]!=null && pp_["background-color"]!=""){background_color_=pp_["background-color"]}
 if((pp_["color"]!=null && pp_["color"]!="")){color_=pp_["color"]}
 var ss_="position: relative;background-color:"+background_color_+";color:"+color_+";width: "+width+"%;height:"+height_+"px;display:block;"
 this.link_content.setAttribute("style", ss_);
 // border: 1px solid #ccc;
 if(is_on_click){
    //alert(this.link_content.outerHTML)
    this.link_content.onclick=container_on_click;
 }

 this.link_content.onchange=container_on_change;
 container.appendChild(this.link_content);
 //alert("9090-2\n"+container.outerHTML)
 this.process_content();
}

TabContent.prototype.process_content = process_content;

TabContent.prototype.set_objects_data = function(dic, is_json_data=false)
{
 this.clear_objects_data()
  //alert("90766-66 "+JSON.stringify(dic))
 // alert(dic["pkf_name"])
 this.is_json_data=is_json_data;

 this.link_content.setAttribute("record_id", dic[dic["pkf_name"]]);
 var container_dic=this.tab.tab_objects[this.link_number];
 for(o in container_dic)
 {
  //alert("90766-66-1 "+JSON.stringify(container_dic[o]["obj_type"]+"\n"+container_dic[o]["obj_name"]))
  var on_=container_dic[o]["obj_name"]
   if(container_dic[o]["obj_type"]=="acObj" && ((on_=="acInput" || on_=="acTextarea" || on_=="acRichText") || on_=="acSelect")){
     var multiple_=false;if("multiple" in container_dic[o]["properties"]){multiple_ = true}
     var eI=getEBI(o);var f_=container_dic[o]["properties"]["field"]
     if(f_==null || f_==""){continue;}
     var v=dic[f_];
     if(v==null){v=""}
      //alert("1 v\n"+v)
      //if(v=="" || v==null){try{eI.value=container_dic[o]["properties"]["default_value"]} catch(er){};continue;}
     //var m=dic[container_dic[o]["properties"]["field"]]
     //if(on_=="acTextarea"){v=JSON.stringify(v);alert(v);}
     //if(on_=="acRichText"){alert(v)}
     if(container_dic[o]["properties"]["field"]=="time_dim")
     {
      var v_=v+"";v_=v_.substr(0, 4)+"-"+v_.substring(4,6)+"-"+v_.substring(6,8)
      //alert(v_)
      eI.value=v_;
      //alert("2 v\n"+v)
     }
     else if (multiple_==true)
     {
       var sv = v.split(',');eI.value = null; // Reset pre-selected options (just in case)
       //alert(JSON.stringify(sv))
       var eILen = eI.options.length;
       for (var i=0;i<eILen;i++) {
       //alert(eI.options[i].value)
        if (sv.indexOf(eI.options[i].value) >= 0) {eI.options[i].selected = true}

       }
     } else if(on_=="acRichText"){
       if(v!=""){eI.innerHTML=v}
     } else {if(v!=""){eI.value=v}}

     var foreign_table=container_dic[o]["properties"]["foreign_table"]
     if(foreign_table!=null && foreign_table!="")
     {
       //alert(JSON.stringify(container_dic[o]["properties"]))
       var field_=container_dic[o]["properties"]["field"]
       if(!this.link_content.foreign_keys){this.link_content.foreign_keys={}}
       this.link_content.foreign_keys[field_]={"value":v, "foreign_table":foreign_table};
     }
     //--
     var dependent=container_dic[o]["properties"]["dependent"]
     if(dependent!=null && dependent!="")
     {
       //alert(JSON.stringify(container_dic[o]["properties"]))
       var field_=container_dic[o]["properties"]["field"]
       if(!this.link_content.dependents){this.link_content.dependents={}}
       this.link_content.dependents[field_]={"value":v, "dependent":dependent};
     }
     //alert(JSON.stringify(this.link_content.foreign_keys))
     //alert(JSON.stringify(this.link_content.dependents))
   }
 }
}

TabContent.prototype.clear_objects_data = function()
{

 try{
   var dic = this.current_record_data;
   this.link_content.setAttribute("record_id", "new");
   var container_dic=this.tab.tab_objects[this.link_number];
   // alert('92 container_dic= ' + JSON.stringify(container_dic));
   this.link_content.foreign_keys={};this.link_content.dependents={};
   var table_=getEBI("content_"+this.link_number)
   //alert(table_.outerHTML)
   var table_=getEBI("content_"+this.link_number).getAttribute("table");
   //alert(table_)
   if(table_==null || table_==""){return}
   for(o in container_dic)
   {
    //alert('95 container_dic= ' + JSON.stringify(container_dic[o]));
     var f_=container_dic[o]["properties"]["field"]
     if(f_==null || f_==""){continue;}

    var obj_name=container_dic[o]["obj_name"];
    if(container_dic[o]["obj_type"]=="acObj" && (obj_name=="acInput" || obj_name=="acTextarea" || obj_name=="acRichText" || obj_name=="acSelect"))
    {if(obj_name=="acSelect"){document.getElementById(o).value=-1}
     else if(obj_name=="acRichText"){getEBI(o).innerHTML="";}
     else {getEBI(o).value="";}
    }
   }
  } catch(er){}
}

TabContent.prototype.delete_record = function()
{
  var id_ = prompt("Are you sure you want to delete this record? if so type Yes" , 'No')
  if(id_ != 'Yes') {return;}
  id_=this.link_content.getAttribute("record_id")
  var table_=this.link_content.getAttribute("table")
  var atm_=this.tab.parent;
  var dic_ = {"obj" : "AdvancedTabs", "atm": atm_.my_name, "app": atm_.my_app, "fun": "delete_record",
              "params": {"app": atm_.my_app, "model": table_, "id":id_}}
              $.post(atm_.activate_function_link_,
              {
                dic : JSON.stringify(dic_)
              },
              function(dic){
                id_ = dic["result"]["id"];
                alert("record was deleted")
             })
}

TabContent.prototype.refresh_my_tables = function(f, v)
{
 //alert("60 TabContent.prototype.refresh_my_tables f="+f+" v="+v)
 var container_dic=this.tab.tab_objects[this.link_number];
 //alert('90 container_dic= '+ JSON.stringify(container_dic));
 for(o in container_dic)
 {
   //alert('90 container_dic= '+ JSON.stringify(container_dic[o]));
   if(container_dic[o]["obj_type"]=="acPlugin" && (container_dic[o]["obj_name"]=="acSearchTable" || container_dic[o]["obj_name"]=="acMSearchTable")){
     try{
          var eI=document.getElementById(o);
          eI.my_creator_obj.search_input_.setAttribute("filter_field", f);
          //alert('9059 eI.my_creator_obj= '+ JSON.stringify(eI.my_creator_obj.parent.data["fields"]));
          var fs=eI.my_creator_obj.parent.data["fields"]
          var n_=0; eI.my_creator_obj.search_input_.value="";
          for(var i in fs)
          {
           if(f==fs[i]["field_name"]){
            var ft_=fs[i]["foreign_table"];
            //alert(f + "  9087  " + ft_)
            if(ft_==null || ft_==""){eI.my_creator_obj.search_input_.setAttribute("filter_field_foreign_table", "")}
            else{eI.my_creator_obj.search_input_.setAttribute("filter_field_foreign_table", ft_)}
            eI.my_creator_obj.search_input_.setAttribute("filter_field", f);
            eI.my_creator_obj.search_input_.setAttribute("placeholder", "Search "+fs[i]["field_title"]+"..");
            eI.my_creator_obj.search_input_.value=v;
            //alert("  9088  " + eI.my_creator_obj.search_input_.outerHTML)
           }
           if(n_==0){
              var ec = new Event("keyup", {bubbles: true});
              //alert('902 eI.my_creator_obj.search_input_.dispatchEvent(ec)= '+ eI.my_creator_obj.search_input_.outerHTML);
              try{eI.my_creator_obj.search_input_.dispatchEvent(ec)} catch(er){alert(er)};n_=1;
            }
          }
        } catch(er){}
   }
 }
}


// -- TabNavLink --
function TabNavLink(tab_nav, link_dic){
 //alert(JSON.stringify(link_dic))
 var pp=tab_nav.properties
 //alert(JSON.stringify(pp))
 this.tab_nav=tab_nav;
 this.tab=tab_nav.tab;
 this.nav=tab_nav.nav;
 this.link_number=link_dic["nav_link"]["properties"]["link_number"];
 this.nav.links[this.link_number]=this;
 this.link=link_dic["nav_link"]["properties"]["link"];
 this.title=link_dic["nav_link"]["properties"]["title"];

 var s_style="width:100%;cursor:pointer";
 var c_ = pp["button_color"];if(c_!=null && c_!=""){var s_style=";color:"+c_}
 var bgc_ = pp["button_bg_color"];if(bgc_!=null && bgc_!=""){var s_style=";background-color:"+bgc_}
 var fw_ = pp["font-weight"];if(fw_!=null && fw_!=""){var s_style=";font-weight:"+fw_}
 var br_ = pp["border_radius"];if(br_!=null && br_!=""){var s_style=";border-radius:"+br_}

 var ba_=pp["button_align"];
 if(ba_!=null && ba_=="vertical")
 {
  this.link_btn=document.createElement("span");
  s_style+=";padding-right: 10px;;padding-top: 10px;padding-bottom: 10px;writing-mode:vertical-rl";
 } else{this.link_btn=document.createElement("button");}

 this.link_btn.innerHTML=this.title;
 //alert(this.link_btn.outerHTML)

 this.link_btn.setAttribute("class","nav_button nav_button"+this.tab.tab_name);
 this.link_btn.setAttribute("id","link_btn_"+this.link_number);
 this.link_btn.setAttribute("obj_type","tab_nav_link");
 this.link_btn.setAttribute("link_number",this.link_number);
 this.link_btn.setAttribute("tab_name",this.tab.tab_name);
 this.link_btn.parent=this;
 this.nav_link_functions=link_dic["nav_link"]["functions"];
 this.tab_content=link_dic["nav_link"]["tab_content"]

 this.link_btn.onclick=function(event){
  //alert(JSON.stringify(this.parent.nav_link_functions))
   try{var f=this.parent.nav_link_functions["onclick"];
     //alert("onclick"+this.parent.link_number+"="+f)
     eval("onclick"+this.parent.link_number+"="+f)
     //alert("onclick"+this.parent.link_number+"(event)")
     var k=eval("onclick"+this.parent.link_number+"(event)")
     try{if(k==-1){return;}} catch(er){alert("Error 555666 TabNavLink: "+er)}
   } catch(er){}
   var e=event.target;
   var tab_name=e.getAttribute("tab_name");

   //e.parent.nav.active_link=this.parent;
   e.parent.nav.set_active_link(this.parent)

   try{for(w in this.parent.tab.PopWinObjects){if(w=="editor"){ continue;};this.parent.tab.PopWinObjects[w].temp_close_win()}} catch(er){}

   try{for(w in this.parent.tab.PopWinObjects){if(w=="editor"){ continue;};
        //alert(this.parent.tab.PopWinObjects[w].popwin_content.link_dic["properties"]["container_id"]+"\n\n"+this.parent.content.link_number)
        if(this.parent.tab.PopWinObjects[w].popwin_content.link_dic["properties"]["container_id"]==this.parent.tab.tab_active_link.content.link_number)
        {this.parent.tab.PopWinObjects[w].resume_win()}
   }} catch(er){}

   if(event.ctrlKey && !event.altKey){
      this.parent.tab.parent.editor.active_link=this.parent;
      this.parent.tab.parent.editor.main_menus["TabNavLink"].btn.click();
      this.parent.tab.parent.editor.sub_menus["TabNavLinkitem"].btn.click();
   }
   var tabcontent=document.getElementsByClassName("tabcontent"+tab_name+"_content");
   for(i=0;i<tabcontent.length;i++){tabcontent[i].style.display='none';}
   var btns=document.getElementsByClassName("nav_button"+tab_name);
   for(i=0;i<btns.length;i++){btns[i].className=btns[i].className.replace(" active","")}
   try{e.parent.link_content.style.display="block";this.parent.link_btn.className+=" active";}catch(er){alert("er9014-11: "+er)}
 }
 for(f in this.nav_link_functions)
 {if(f != "onclick"){var fun=this.nav_link_functions[f]; eval("this.link_btn."+f+"="+fun)}}

 this.nav.appendChild(this.link_btn);

 this.link_btn.setAttribute("style", s_style);
 this.nav.links[this.link_number]=this;
 //this.tab_nav
 //.nav_one_tab_content=
 if(this.tab.tab_nav_links["properties"]["nav_type"]=="navmulti"){
   this.content=new TabContent(this.tab, tab_nav.nav_content, link_dic["tab_content"], is_on_click=true, is_link=true);
 } else if (this.tab.tab_nav_links["properties"]["nav_type"]=="navone")
 {
   if(this.tab_nav.nav_one_tab_content == null)
   {
    this.tab_nav.nav_one_tab_content=new TabContent(this.tab, tab_nav.nav_content, link_dic["tab_content"], is_on_click=true, is_link=true);
   }
   this.content=this.tab_nav.nav_one_tab_content;
 }
//alert(994)
 this.link_content=this.content.link_content;
 this.link_content.className+=this.tab.tab_name+"_content";
}

// -- TabNav --
function TabNav(tab_=null){
  //alert(JSON.stringify(tab_.tab_nav_links));
  this.tab=tab_;
  this.nav=null;
  this.active_link=null;
  this.nav_one_tab_content=null;
  this.properties=this.tab.tab_nav_links["properties"];
      //alert("AAA"+JSON.stringify(this.properties));
  this.is_show_btns=this.properties["is_show_btns"];
  this.nav_width=this.properties["width"];
  this.add_title=this.properties["add_title"];
  this.remove_title=this.properties["remove_title"];
  this.create_main_tab_nav_content();
  this.process_data();
}

TabNav.prototype.create_main_tab_nav_content = function()
{
   this.nav_container=document.createElement("div");
   this.nav_container.editor=this.tab.parent.editor;
   this.nav_container.setAttribute("my_name", "TabNav.nav_container_"+this.properties["obj_number"]);
   this.nav_container.setAttribute("id", this.properties["obj_number"]);
   if("left" in this.properties){var left_=this.properties["left"];} else {var left_="0"}
   if("top" in this.properties){var top_=this.properties["top"];} else {var top_="0"}
   //border:1px solid #ccc;background-color: "+this.properties["background_color"]+";
   var style_ = "position: relative;z-index:10000;left:"+left_+"px;top:"+top_+"px;width:"+this.nav_width+"%;"
   this.nav_container.setAttribute("style", style_);
   // --
   this.nav = document.createElement("div");
   for(f in this.tab.tab_nav_links["functions"])
   {
     eval(f+"_zz"+this.properties["obj_number"]+"="+this.tab.tab_nav_links["functions"][f]);
     eval('this.nav.'+f+'='+f+'_zz'+this.properties["obj_number"]);
   }
   this.nav.links={};
   this.nav.my_creator_obj=this;
   //--
   this.nav.set_active_link=function(link_obj){this.active_link=link_obj;this.my_creator_obj.tab.tab_active_link=link_obj;
   // alert(this.active_link.link_number);
   }

   //--
   if(this.is_show_btns=="true")
   {
     this.nav_container_buttons=document.createElement("div");
     this.nav_btn_add=document.createElement("button");
     this.nav_container.appendChild(this.nav_container_buttons)
     this.nav_container_buttons.appendChild(this.nav_btn_add)
     this.nav_btn_add.parent=this;
     this.nav_btn_add.innerHTML=this.add_title;
     this.nav_btn_add.onclick=function(event){
         //-
         var link_ = prompt("Enter name for new link:",'');if(link_==''){alert("Please enter a link name");return}
         var link_number=this.parent.tab.get_next_obj_number();
         var width_=100-this.parent.nav_width;

         var link_dic={"nav_link":{"properties":{"link_number":link_number, "link":link_, "title":link_},
                                   "functions":{}},
                       "tab_content":{"properties":{"link_number":link_number, "content_type": "simple", "width":width_,
                                                    "table":""},
                                      "functions":{}}
                       };

         this.parent.tab.tab_objects[link_number]={};this.parent.tab.tab_nav_links["nav_links"][link_number]=link_dic;
         //--
         var tab_nav_links = new TabNavLink(this.parent, link_dic);
         //--
         tab_nav_links.link_btn.click();
         this.parent.tab.parent.save();
         this.parent.nav_content.appendChild(tab_nav_links.link_content);
         this.parent.nav.appendChild(tab_nav_links.link_btn);
         this.parent.nav.links[link_number]=tab_nav_links;
         tab_nav_links.link_btn.click();
         //alert(JSON.stringify(this.parent.tab.tab_nav_links))
         this.parent.tab.parent.save();
     }

     this.nav_btn_remove=document.createElement("button");
     this.nav_container_buttons.appendChild(this.nav_btn_remove)
     this.nav_btn_remove.parent=this;
     this.nav_btn_remove.innerHTML=this.remove_title;
     this.nav_btn_remove.onclick=function(event){
         //-
         var e=event.target;
         //alert(e.outerHTML)
         //alert(this.parent.nav.active_link)
         var link_= this.parent.nav.active_link.link_btn.innerHTML;
         var link_number= this.parent.nav.active_link.link_btn.getAttribute("link_number")
         var link_ = prompt("Are you sure you want to remove link "+link_+"? if so type Yes:",'No');
         if(link_=='Yes')
         {
          delete this.parent.tab.tab_nav_links["nav_links"][link_number];
          this.parent.nav.active_link.link_btn.outerHTML="";
          this.parent.nav.active_link.link_content.outerHTML="";
          delete this.parent.nav.links[link_number];
          this.parent.tab.parent.save();
         }
     }
   }
   this.nav_container.appendChild(this.nav)
   this.tab.content.appendChild(this.nav_container)
   this.nav_content = document.createElement("div");
   this.nav_content.setAttribute("my_name", "TabNav.nav_content");
   var width_=100-this.nav_width;
   this.nav_content.setAttribute("style","float:left;background-color:#f1f1f1;width:"+width_+"%;height:100%;");
   // border:1px solid #ccc;
   this.tab.content.appendChild(this.nav_content)
   this.nav.setAttribute("style","float:left;background-color:"+this.properties["background_color"]+";width:100%;");
}

TabNav.prototype.process_data = function()
{
 var n1_obj=null;
 for (n_ in this.tab.tab_nav_links["nav_links"])
 {
  var link_dic=this.tab.tab_nav_links["nav_links"][n_];
//--
  var nav_link_obj=new TabNavLink(this, link_dic);
//--
  if(n1_obj==null){n1_obj=nav_link_obj.link_btn}
 }
 try{n1_obj.click();} catch(er){}
}


// Tab obj ---
function Tab(parent, data, id)
{
 //alert("Tab");
 //alert("id="+id+"\n"+JSON.stringify(data));
 this.n_max_zindex=0
 this.parent=parent; this.tab_id=id;
 this.tab_active_link=null;
 this.btn=null;this.content=null;this.PopWinObjects={};
 this.tab_properties=data["properties"];

 if(this.tab_properties["tab_type"].includes("nav")) {this.is_on_click=false;} else {this.is_on_click=true;}

 this.link_number=data["properties"]["link_number"];
 this.tab_name=data["properties"]["tab_name"];
 this.tab_order=data["properties"]["tab_order"];
 this.tab_title=data["properties"]["tab_title"];
 this.tab_type=data["properties"]["tab_type"];
 //--
 if(!("functions" in data)){this.tab_functions={};
   this.tab_functions[this.tab_name+"__init__"]=this.tab_name+"__init__=function(called_tab, calling_tab){\ntry{\n\n} catch(er){alert('er9020: '+ er)}}";
   this.tab_functions[this.tab_name+"__myclick__"]=this.tab_name+"__myclick__=function(called_tab, calling_tab){\ntry{\n\n} catch(er){alert('er9021: '+ er)}}";
   this.tab_functions[this.tab_name+"__otherclick__"]=this.tab_name+"__otherclick__=function(called_tab, calling_tab){\ntry{\n\n} catch(er){alert('er9022: '+ er)}}";
  } else {this.tab_functions=data["functions"]
 //alert(JSON.stringify(this.tab_functions=data["functions"]));
 };
 //--
 if(!("tab_pop_win_buttons" in data)){this.tab_pop_win_buttons={"properties":{},"pop_wins":{}}} else
 {this.tab_pop_win_buttons=data["tab_pop_win_buttons"]}
 //--
 if(!("tab_content_link_dic" in data))
 {this.tab_content_link_dic={"properties":{"link_number":this.link_number, "content_type": "simple", "width":100,
                             "table":"", "parent_table":""}, "functions":{} }}
 else {this.tab_content_link_dic=data["tab_content_link_dic"]}
 //--
 if(!("objects" in data)) {this.tab_objects={};this.tab_objects[this.link_number]={};} else {this.tab_objects=data["objects"]};
 this.tab_objects_created={}
 //alert(JSON.stringify(this.tab_objects));
 this.create_btn_container(data);
 //alert(JSON.stringify(this.tab_objects))
 this.process_functions();
 this.init_tab();
 //this.process_content();
 this.new_obj_to_create = null;
 this.active_obj=null;
 this.is_int_objects_data=false;
}

Tab.prototype.int_objects_data = function()
{
 if(this.is_int_objects_data==true){return};
 for(var z in this.tab_objects_created){
   try{var o=this.tab_objects_created[z];
       try{if(o.creator.is_call_get_data!="false"){
        try{o.creator.get_data();} catch (er){}};
       o.is_call_get_data=true} catch(er){}
      } catch(er){}
   }
 this.is_int_objects_data=true;
}

Tab.prototype.create_btn_container = function(data)
{
 //alert(JSON.stringify(data["properties"]));
  var pp=data["properties"]
  this.btn=document.createElement("button");
  if(this.tab_name=="main_panel"){
   var titles_height=50
   atm.titles_padding = 10
   atm.titles_width=170;
   var titles_background_color = "green"
   atm.titles_height=titles_height-2*this.titles_padding;
   var style_="text-align:center;background-color:"+titles_background_color+";height:"+titles_height+"px;padding:5px;";
   atm.titles.setAttribute("style", style_)
  }
  var s_style="cursor:pointer;";var bgc=""; var width_=""
  if(atm.titles_height!=0){
    s_style+="height:"+atm.titles_height+"px";
    bgc=this.parent.titles.style.backgroundColor;
    width_=atm.titles_width;
    s_style+=";border-bottom:none;border-top:none;border-left:none;border-right:2px solid white;color:white;padding:"+atm.titles_padding+"px 32px "+atm.titles_padding+"px;text-align:center;text-decoration:none;" //display:inline-block;"
    s_style+="font-size: 16px;margin: 4px 2px"
  }
   if(("button_bg_color" in pp) && (pp["button_bg_color"]!="#ffffff")){bgc=pp["button_bg_color"]};
   if(bgc!=""){s_style+=";background-color:"+bgc;}
   if(("button_color" in pp) && (pp["button_color"]!="")){s_style+=";color:"+pp["button_color"]}
   if(("font-weight" in pp) && (pp["font-weight"]!="")){s_style+=";font-weight:"+pp["font-weight"]}
   if(("border_radius" in pp) && (pp["border_radius"]!="")){s_style+=";border-radius:"+pp["border_radius"]}
   if(("width" in pp) && (pp["width"]!="")){width_=pp["width"]};
   if(width_!=""){s_style+=";width:"+width_+"px"}

  this.btn.setAttribute("style", s_style);

  this.btn.setAttribute("link_number", this.link_number);
  this.btn.parent=this;
  this.btn.setAttribute("class", "tablinks");
  this.btn.className+=" active";
  this.btn.innerHTML=this.tab_title;
  this.btn.onclick=function(event)
  {
    try{var btn=event.target;btn.parent.parent.set_active_tab(btn);
    this.parent.int_objects_data();
    } catch(er) {alert("Error 22: "+er)}
  }


  if((atm.dic["is_show_btns"] == false) && (this.tab_name=="main_panel")){} else{
    this.parent.titles.appendChild(this.btn)
  }
  //--
  //if(this.tab_properties["tab_type"].includes("nav")) {var is_on_click=false;} else {var is_on_click=true;}
  if(this.tab_name=="main_panel"){
     this.tab_content_link_dic["properties"]["height"]=pp["height"]
     this.tab_content=new TabContent(tab=this, container=this.parent.panel, link_dic=this.tab_content_link_dic, is_on_click=this.is_on_click, is_link=false);
  } else{
     this.tab_content=new TabContent(tab=this, container=this.parent.container, link_dic=this.tab_content_link_dic, is_on_click=this.is_on_click, is_link=false);
  }

  this.content=this.tab_content.link_content;
  this.content.parent=this;
  if(this.tab_properties["tab_type"].includes("nav"))
  {
    // alert(JSON.stringify(data));
    // alert(JSON.stringify(data["nav_links"]));
     if(!("nav_links" in data) || data["nav_links"]=="null")
     {
       var n_=this.get_next_obj_number();
       this.tab_nav_links={"properties":{"obj_number":n_, "nav_type":this.tab_properties["tab_type"],"width":"10",
                                         "add_title":"Add", "remove_title":"Remove",
                                         "is_show_btns":"true", "background_color": "#f1f1f1"},
                           "functions":{},
                           "nav_links":{}};
     } else {this.tab_nav_links=data["nav_links"]};
     this.tab_nav=new TabNav(tab=this);
  }
}

Tab.prototype.create_tab_pop_wins = function(){
try{
   for(var i in this.tab_pop_win_buttons["pop_wins"]){
       var dic_=this.tab_pop_win_buttons["pop_wins"][i]
       var win_obj=this.get_pop_win_obj(dic_);
       win_obj.__init__();
       var pp_=dic_["properties"]
       win_obj.set_win_frame_style(pp_["zindex"], pp_["height"], pp_["width"], pp_["right"], pp_["top"], pp_["background_color"])
       win_obj.set_acWinStatEventListeners(this.parent.editor);       //win_obj.set_acWinStat('block');
       win_obj.resume_win()
     }
   } catch(er) {alert('er9024: '+ er)}
}

Tab.prototype.process_functions = function(){for(f in this.tab_functions){eval(this.tab_functions[f])}}

Tab.prototype.init_tab=function(){try{eval(this.tab_name+"__init__(called_tab=this, calling_tab=this)");} catch(er) {}}

Tab.prototype.get_next_obj_number=function(){return this.parent.get_next_obj_number();}

Tab.prototype.set_to_create_obj=function(dic){
 //alert("set_to_create_obj")
 //alert(JSON.stringify(dic));
 this.new_obj_to_create = dic;
 this.parent.new_obj_to_create=null;
}

Tab.prototype.generate_obj=function(dic=null){
 //alert("generate_obj")
 //alert(JSON.stringify(dic));
 var s='var obj = this.parent.get_obj(this, dic);'
 //alert(s)
 try{eval(s)} catch(er){alert("er200: "+er)}
 obj.create_obj();
 this.tab_objects_created[dic["properties"]["obj_number"]]=obj;
 // obj.create_editor();
 //alert(JSON.stringify(obj.data));
 this.new_obj_to_create = null;
 return obj;
}

Tab.prototype.set_max_zindex = function(win) {

  var number = win.win_frame.style.zIndex;
  if(number==""){number=0}
  if(1*number>1*this.n_max_zindex){this.n_max_zindex=1*number+1} else{
  this.n_max_zindex=1*this.n_max_zindex+1}
  win.win_frame.style.zIndex=this.n_max_zindex
     var win_number=win.win_frame.getAttribute("win_number")
     if(win_number!=null){
          var tab_id=win.win_frame.getAttribute("tab_id")
          var win_name=win.win_frame.getAttribute("my_name");
     //alert("this.n_max_zindex "+this.n_max_zindex+" win_number "+ win_number+" tab_id "+ tab_id+" win_name "+ win_name)

    try{
        if(win_name!="editor"){
          var ret=win.create_editor_for_popwin(win.tab_obj_.tab_pop_win_buttons["pop_wins"][win_number]);
          if(ret==1){this.parent.editor.active_popup_win=[tab_id, win_name, win_number]}
        }
    } catch(er){
     alert("error505:\n"+er+"\nwin_name="+win_name+"\nwin_number="+win_number)
    }
  }
}

Tab.prototype.get_pop_win_obj = function(dic)
{
 //alert("90602-11 get_pop_win_obj\n" + JSON.stringify(dic));
 //alert("90602-22 get_pop_win_obj\n" + JSON.stringify(dic["properties"]));
 //alert("90602-33 get_pop_win_obj\n" + JSON.stringify(dic["popwin_content"]));
 var pp_=dic["properties"];
 var s_name=pp_["name"]; var s_title=pp_["title"];
 var title_color=pp_["title_color"];var title_background_color=pp_["title_background_color"];
 var link_number =pp_["id"];
 var s = 'function TabPopWin'+this.tab_name+s_name+'(parent,dic_)';
 s+='{'
  // s+='alert(JSON.stringify(dic_));'
  // s+='alert(parent.tab_name);'

 s+='this.my_name="'+this.tab_name+s_name+'";';
 s+='this.name="win_'+this.tab_name+s_name+'";';
 s+='var is_scroll_=true;';
 s+='this.atm = parent.parent;'
 s+='this.main_menus = {};this.sub_menus = {};'

 s+='acWin.call(this,my_name_=this.my_name, win_name=this.name, win_title="'+s_title+'",';
 s+='right= "2%", top="30%",'
 s+='is_scroll=is_scroll_, zindex="21", tab_obj_=parent, is_nav_panel=true, win_number='+link_number+');'
 s+='this.tab_obj_.tab_pop_win_buttons["pop_wins"]['+link_number+']=dic_;'

 //s+='alert("90699-inner\\n "+JSON.stringify(this.tab_obj_.tab_pop_win_buttons["pop_wins"]['+link_number+']));'
 //alert("link_number:"+link_number)

 s+='if(!("'+link_number+'" in this.tab_obj_.tab_objects)){this.tab_obj_.tab_objects["'+link_number+'"]={}};'
 s+='this.popwin_content=new TabContent(tab=parent, container=this.win_content, link_dic=dic_["popwin_content"], is_on_click=true, is_link=false);'
 s+='this.win_content=this.popwin_content.link_content;'
 s+='this.win_content.setAttribute("my_type", "TabPopWin");';
 //s+='this.popwin_content.link_content.my_creator_obj=this;'
 s+='this.win_content.setAttribute("class", "tabcontent"+parent.tab_name);'
 s+='this.win_content.style.display="block";'
 s+='this.win_content.parent=this;'
 s+='};';
 //alert("90605\n" + s);
 eval(s);

 s = 'TabPopWin'+this.tab_name+s_name+'.prototype = Object.create(acWin.prototype);';
 eval(s);
 s = 'TabPopWin'+this.tab_name+s_name+'.prototype.constructor = TabPopWin'+this.tab_name+s_name;
 eval(s);
 //alert("90607\n" + s);

 s='TabPopWin'+this.tab_name+s_name+'.prototype.create_editor_for_popwin = function(dic)'
 s+='{'
 //s+='alert(this.tab_obj_.parent);'
 //s+='alert(this.tab_obj_.parent.editor);'
 //s+='alert("88888888 create_editor_for_popwin: "+JSON.stringify(dic));'

 s+='var tab_=this.tab_obj_.parent.tabs[dic["properties"]["tab_id"]];';
 s+='var pop_win=tab_.PopWinObjects[dic["properties"]["name"]];'
 s+='var pop_win_dic_=tab_.tab_pop_win_buttons["pop_wins"][dic["properties"]["id"]];'
 //s+='alert(100);alert(JSON.stringify(pop_win_dic_));'
 //s+='this.fpe=new FunctionsPropertiesEditor(tab_,pop_win_dic_["functions"],pop_win_dic_["functions"],pop_win_dic_["properties"],pop_win_dic_["properties"]);'
 s+='var pop_win_content=tab_.parent.pop_win;'
 s+='if(this.tab_obj_.parent.editor!=null){'
 //s+='alert(this.tab_obj_.parent.editor.get_functions_properties_editor);'
 s+='this.fpe=this.tab_obj_.parent.editor.get_functions_properties_editor(tab_,pop_win_dic_["functions"],'
 s+='pop_win_content["functions_list"],'
 s+='pop_win_dic_["properties"],'
 s+='pop_win_content["settings_list"],'
 s+='pop_win_content["attributes_list"],'
 s+='tab_btn_name="PopWin",null,node_to_delete=".tab_pop_win_buttons[\'pop_wins\']["+dic[\'properties\'][\'id\']+"]");return 1;'
 s+='} else {return -1};'
 s+='}'

 //alert("90611\n" + s);
 eval(s);
 //alert("90611-2\n" + s);
 //--
 s='TabPopWin'+this.tab_name+s_name+'.prototype.get_my_tab = function()'
 s+='{'
 s+='return this.tab_obj_.parent.tabs['+pp_["tab_id"]+'];';
 s+='}'
 //alert("90615\n" + s);
 eval(s);
 //--
 //for(f in dic['functions']) {s_= f+"_"+pp_["id"]+ '='+dic['functions'][f];alert(s_);eval(s_);}
 //--
 s='TabPopWin'+this.tab_name+s_name+'.prototype.__init__ = function()'
 s+='{'
 s+='this.set_tab(this.tab_obj_);this.set_title_colors("'+title_color+'", "'+title_background_color+'");'
 s+='this.tab_obj_.PopWinObjects[this.my_name]=this;'
 s+='this.id='+link_number+';'
 s+='this.set_title(this.win_title_);'
 s+='dic_fs=this.tab_obj_.tab_pop_win_buttons["pop_wins"]['+link_number+']["functions"];'

 //s+='alert(JSON.stringify(dic_fs));'
 s+='for(f in dic_fs) {var s_= "this."+f+"'+link_number+'="+dic_fs[f];eval(s_);};'
 s+='try{this.__init___'+link_number+'(this)} catch(er){alert("er90358: "+ er)};';
 if(pp_["is_panel"]=="true")
 {
  s+='this.main_menu = document.createElement("div");'
  s+='this.sub_menu = document.createElement("div");'
  s+='this.win_nav_panel.appendChild(this.main_menu);'
  s+='this.win_nav_panel.appendChild(this.sub_menu);'
  s+='try{this.__get_panel_structure___'+link_number+'(this)} catch(er){alert("er9033: "+ er)};';

  //s+='alert(JSON.stringify(this.buttons));'

  s+='var buttons_=this.buttons;'
  s+='for (b in buttons_){'
  s+=' eval("MenuBtnWin"+b+"=this.get_main_button_win_obj(b, buttons_[b][\'width\'], buttons_[b][\'title\'], buttons_[b][\'sub_buttons\'], buttons_[b][\'obj_type\'])");'

  //s+=' alert(this.name+"--1-- "+b);'
  s+=' eval("var nbw=new MenuBtnWin"+b+"(parent=this)");'
  s+=' nbw.btn.is_run_first_sub_menu=false;nbw.btn.click();nbw.btn.is_run_first_sub_menu=true'
  s+='};'
  s+='try{this.__set_panel___'+link_number+'(this)} catch(er){alert("er9026: "+ er)};';
 }
 s+='}'
 //alert("90620\n" + s);
 eval(s);

 // ---
 //alert("90620-1\n");
 // this.set_panel();
 // this.main_menus["Tab"].btn.click()
 //alert(eval('TabPopWin'+this.active_tab.tab_name+s_name+'.prototype.set_title_colors'))
 var tab_id_=dic["properties"]["tab_id"]

 //alert(JSON.stringify(this.parent.tabs));
 var tab__=this.parent.tabs[tab_id_]
 s='new TabPopWin'+this.tab_name+s_name+'(tab__, dic)';
 //alert("90677\n" + s);
 try{
 var result_obj= eval(s)
 } catch (er) {alert("er4550: "+er)}
 this.parent.pop_wins[link_number]=result_obj
 return result_obj;
}

// -- acWin popup window --
function acWin(my_name_="none", win_name="none", win_title="none", right= "0%", top="0%", is_scroll=true, zindex="11",
               tab_obj_=null, is_nav_panel=false, win_number=0)
{
  //alert(7)
  //alert(tab_obj_.tab_name)
  //alert(win_name)
  //alert(my_name_)
  //create its div for window
  this.win_number=win_number;
  this.tab_obj_=tab_obj_;
  this.is_nav_panel = is_nav_panel;
  this.nav_height = 0;
  this.is_scroll = is_scroll
  this.win_name = win_name
  this.my_name_ = my_name_;
  this.win_title_ = win_title;
  this.win_frame = document.createElement("div");
  this.win_frame.setAttribute("win_number", this.win_number);
  this.win_frame.setAttribute("tab_id", this.tab_obj_.tab_id);
  this.win_frame.setAttribute("id", "win_frame_"+this.win_name);
  this.win_frame.setAttribute("my_name", my_name_);
  // TITLE for window
  this.title_height = 25
  this.win_frame_title = document.createElement("div");
  this.win_frame_title.setAttribute("id", "win_frame_title_"+this.win_name);
  // --
  this.win_frame_ico = document.createElement("img");
  this.win_frame_ico.setAttribute("id", "win_frame_ico_"+this.win_name);
  this.win_frame_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_frame_ico.setAttribute("style", "border-radius: 10px;position: absolute;left: 45px");
  this.win_frame_ico.setAttribute("height", "20");
  this.win_frame_ico.setAttribute("width", "20");
  // --
  this.win_frame_title.appendChild(this.win_frame_ico)
  this.win_frame_title_span = document.createElement("span");
  this.win_frame_title_span.innerHTML = win_title
  this.win_frame_title.appendChild(this.win_frame_title_span)
  this.win_frame_msg_span = document.createElement("span");
  this.win_frame_title.appendChild(this.win_frame_msg_span);

  this.win_frame_space_span = document.createElement("span");
  this.win_frame_space_span.innerHTML = "&nbsp;&nbsp;";
  this.win_frame_title.appendChild(this.win_frame_space_span);

  this.win_frame_right_span = document.createElement("span");
  this.win_frame_title.appendChild(this.win_frame_right_span);
  this.win_frame.appendChild(this.win_frame_title)
  // NAV PANEL --
  if(this.is_nav_panel==true)
  {
   this.nav_height = 25;
   this.win_nav_panel = document.createElement("div");
   this.win_nav_panel.setAttribute("id", "win_nav_panel_"+this.win_name);
   var nav_panel_background_color = "lightgray";
   this.style_nav_panel = "background-color: "+nav_panel_background_color+";"
   this.style_nav_panel += "width:100%;font-size:12pt;"
   this.style_nav_panel += "border:2px solid #0094ff;"
   this.win_nav_panel.setAttribute("style", this.style_nav_panel);
   this.win_frame.appendChild(this.win_nav_panel)
  }
  // CONTENT --
  this.win_content = document.createElement("div");
  this.win_content.setAttribute("id", "win_content_"+this.win_name);
  // -- call it from outside --
  //this.set_win_frame_style(zindex, height="300px", width="300px", right, top, "white");
  //--
  this.win_frame.appendChild(this.win_content)

  //-- Little div TO CLOSE and OPEN the WIN --
  this.win_nav = document.createElement("div");
  this.win_nav.setAttribute("id", "win_nav_"+this.win_name);
  this.style_win_nav = "position:fixed;text-align:center;width:10px;display:none; right:"+right+";top:"+top+";";
  this.win_nav.setAttribute("style", this.style_win_nav)
  // --
  this.win_nav_ico = document.createElement("img");
  this.win_nav_ico.setAttribute("id", "win_nav_ico_"+this.win_name);
  this.win_nav_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_nav_ico.setAttribute("style", "border-radius: 10px;position: absolute;right:97%");
  //--
  this.win_nav_ico.setAttribute("height", "5");
  this.win_nav_ico.setAttribute("width", "5");
  //
  this.win_nav.appendChild(this.win_nav_ico)
  // --
  this.span = document.createElement("span");
  this.span.setAttribute("id", "span_"+this.win_name);
  //this.span.innerHTML = "&nbsp;Whiteboard"
  this.span.setAttribute("style", "position: absolute;left: 50");
  this.win_nav.appendChild(this.span)
  // --
  this.body_ = tab_obj_.parent.elm_body;
  this.body_.appendChild(this.win_frame);
  this.body_.appendChild(this.win_nav);
  // --
  this.win_frame_title.setAttribute('pos1', 0);
  this.win_frame_title.setAttribute('pos2', 0);
  this.win_frame_title.setAttribute('pos3', 0);
  this.win_frame_title.setAttribute('pos4', 0);
  this.win_frame.style.display = "none"
  this.win_frame_style_display = this.win_frame.style.display;
}

acWin.prototype.set_tab = function(tab_obj){this.tab_obj_=tab_obj;}

acWin.prototype.set_title_colors = function(color, b_color)
{
  this.style_title = "text-align:center;display:block;padding:4px;color:"+color+";margin:-1 0 0 0;height:"+this.title_height+"px;z-index:11;"
  this.style_title += "background-color:"+b_color+";cursor:move;font-size:12pt;border-top-left-radius:35px;border-top-right-radius:35px;"
  this.win_frame_title.setAttribute("style", this.style_title);
}
acWin.prototype.set_win_frame_style = function(zindex, height, width, right, top, background_color)
{
  this.style_frame = "position: fixed; background-color: "+background_color+"; z-index: "+zindex+"; text-align: left;"
  this.style_frame += "height:"+height+"px;width:"+width+"px;font-size:12pt;"
  this.style_frame += "border:2px solid #0094ff;"
  this.style_frame += "border-top-left-radius:35px;border-top-right-radius:35px;"
  this.style_frame += "right:"+right+";top:"+top+";"
  this.style_frame += "display:block;";
  this.win_frame.setAttribute("style", this.style_frame);
  this.style_content = "height:"+(height-this.title_height-7-this.nav_height)+"px;width:"+width+"px;"
  if(this.is_scroll==true){this.style_content += "overflow: scroll;display:block;"}
  this.win_content.setAttribute("style", this.style_content);
}
acWin.prototype.set_acWinStat = function(ss)
{
    //open and close acWin.
    //alert("90678\n"+ss+"\n"+this.win_frame.style.display)
    if (ss != this.win_frame.style.display)
    {
    try{
      var event_close_ac = new Event("click", {bubbles: true});
      this.win_frame_ico.dispatchEvent(event_close_ac);
      this.win_frame_style_display = ss;
      } catch(er){alert("error 77")}
    }
    this.tab_obj_.set_max_zindex(this)
}
acWin.prototype.close_win = function(){this.set_acWinStat('none');}
acWin.prototype.temp_close_win = function(){this.win_frame.style.display = "none";}
acWin.prototype.resume_win = function(){this.win_frame.style.display = this.win_frame_style_display;}
acWin.prototype.set_title = function(title){this.win_frame_title_span.innerHTML=title;this.set_acWinStat('block');}
acWin.prototype.set_msg = function(msg){this.win_frame_msg_span.innerHTML=msg;} //this.set_acWinStat('block');
acWin.prototype.set_right_msg = function(msg){this.win_frame_right_span.innerHTML=msg;} //this.set_acWinStat('block');
acWin.prototype.get_msg = function(){return this.win_frame_msg_span.innerHTML;}
acWin.prototype.set_info = function(content){this.win_content.innerHTML=content;}
acWin.prototype.add_info = function(content){this.win_content.innerHTML+=content;}
// not used to be deleted.
//acWin.prototype.remove_win = function(){try{this.win_frame.outerHTML="";this.win_nav.outerHTML="";} catch (err) {alert(err);console.log("Error remove_win :", "remove_win 0450");}}
acWin.prototype.move_frame = function(pos1, pos2, pos3, pos4)
{
  // console.log("acWin.prototype.move_frame: ", pos1, pos2, pos3, pos4)
  this.win_frame_title.setAttribute("pos3", pos3);
  this.win_frame_title.setAttribute("pos4", pos4);
  this.win_frame.style.top = (this.win_frame.offsetTop - pos2) + "px";
  this.win_frame.style.left = (this.win_frame.offsetLeft - pos1) + "px";
  this.win_nav.style.top = this.win_frame.style.top;
}
acWin.prototype.set_acWinStatEventListeners = function(ss_obj)
{
  //console.log("set_acWinStatEventListeners :", "Check 0000");
  //alert("var " +  ss_obj.my_name + '= ss_obj')

  // Do we need this??
  //eval("var " +  ss_obj.my_name + '= ss_obj')

  var s = ''
  s += 'this.win_frame_ico.addEventListener("click", function(){'
  s += '  event.preventDefault();'
  s += 'var my_name = x_'+this.win_name+'.getAttribute("my_name");'
  s += '  if (x_'+this.win_name+'.style.display === "none") {'
  s += '    x_'+this.win_name+'.style.display = "block";'
  s += '    n_'+this.win_name+'.style.display = "none";'
  s += '  } else {'
  s += '    x_'+this.win_name+'.style.display = "none";'
  s += '    n_'+this.win_name+'.style.display = "block";'
  s += '  };'
  s += '}.bind(elm = this.win_frame_ico, event, x_'+this.win_name+' = this.win_frame, n_'+this.win_name+' = this.win_nav));'

  try{
   eval(s)
   //console.log("Good :set_acWinStatEventListeners: ", "eval 100")
   } catch (err) {console.log("Error 101", err.message)}
  //console.log("set_acWinStatEventListeners :", "Check 0100");
  s = ''
  s += 'this.win_nav_ico.addEventListener("click", function(){'
  s += 'event.preventDefault();'
  s += 'var my_name = x_'+this.win_name+'.getAttribute("my_name");'
  s += 'if (x_'+this.win_name+'.style.display==="none") {'
  s += 'x_'+this.win_name+'.style.display="block";'
  s += 'n_'+this.win_name+'.style.display="none";'
  s += '} else {'
  s += 'x_'+this.win_name+'.style.display="none";'
  s += 'n_'+this.win_name+'.style.display="block";'
  s += '}'
  s += '}.bind(this.win_nav_ico, event, x_'+this.win_name+' = this.win_frame, n_'+this.win_name+' = this.win_nav));'
  try{
   //console.log("eval 200:"+ss_obj, s)
   eval(s)
   //console.log("Good :set_acWinStatEventListeners: ", "eval 200")
   } catch (err) {console.log("Error 201", err.message)}
  //console.log("set_acWinStatEventListeners :", "Check 0200");

  s = '';
  s += 'this.win_frame_title.addEventListener("mousedown", function(){'
  //s += 'alert(this.outerHTML);'
  //s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos3", event.clientX);'
  s += 'event.preventDefault();'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos3", event.clientX);'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos4", event.clientY);'

  s += 'document.addEventListener("mouseup", my_mouse_up_'+this.win_name+' = function(){'
  s += 'event.preventDefault();'
  s += 'try{'
  s += 'document.removeEventListener("mouseup", my_mouse_up_'+this.win_name+');'
  s += 'document.removeEventListener("mousemove", my_mouse_move_'+this.win_name+');'
  s += '} catch (err) {err.message}'
  s += ' tab_obj__.set_max_zindex(win_'+this.win_name+');'
  s += '}.bind(elm_win_frame_title_'+this.win_name+', event, tab_obj__, win_'+this.win_name+', elm_win_frame_' +this.win_name + '));'
  s += 'document.addEventListener("mousemove", my_mouse_move_'+this.win_name+' = function(){'
  s += 'event.preventDefault();'
        // calculate the new cursor position:
  s += 'pos3 = elm_win_frame_title_'+this.win_name+'.getAttribute("pos3");'
  s += 'pos4 = elm_win_frame_title_'+this.win_name+'.getAttribute("pos4");'
  s += 'pos1 = pos3 - event.clientX;'
  s += 'pos2 = pos4 - event.clientY;'
  s += 'pos3 = event.clientX;'
  s += 'pos4 = event.clientY;'
  s += 'var msg_ = "' + this.my_name_ + '" + ","+pos1+","+pos2+","+pos3+","+pos4;'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos3", pos3);'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos4", pos4);'
        // set the element's new position:
  s += 'elm_win_frame_'+this.win_name+'.style.top = (elm_win_frame_'+this.win_name+'.offsetTop - pos2) + "px";'
  s += 'elm_win_frame_'+this.win_name+'.style.left = (elm_win_frame_'+this.win_name+'.offsetLeft - pos1) + "px";'
  s += 'elm_win_nav_'+this.win_name+'.style.top = elm_win_frame_'+this.win_name+'.style.top;'
  s += 'elm_win_nav_'+this.win_name+'.style.right = elm_win_frame_'+this.win_name+'.style.right;'
  s += '}.bind(elm_win_frame_title_'+this.win_name+', event, elm_win_frame_'+this.win_name+', elm_win_nav_'+this.win_name+'));'
  s += '}.bind(elm_win_frame_title_'+this.win_name+' = this.win_frame_title, event, elm_win_frame_'+this.win_name
  s += ' = this.win_frame, elm_win_nav_'+this.win_name+' = this.win_nav, tab_obj__=this.tab_obj_, win_'+this.win_name+'=this));'

  try{
   //console.log("eval 300:", s)
   eval(s)
   //console.log("Good :set_acWinStatEventListeners: ", "eval 300")
   } catch (err) {alert("Error 300" + err.message)}
  //console.log("set_acWinStatEventListeners :", "Check 0300");
  try{
        var event_frame_ico = new Event("click", {bubbles: false});
        this.win_frame_ico.dispatchEvent(event_frame_ico);
    } catch (err) {alert("error 99");console.log("Error set_acWinStatEventListeners :", "Check 0350");}

  // console.log("set_acWinStatEventListeners :", "Check 0400");
}

acWin.prototype.get_main_button_obj = function(s_name)
{
  try{ return this.main_menus[s_name] }catch(er){alert(er)}
  return null;
}

acWin.prototype.get_sub_menus_button_obj = function(s_name)
{
  try{ return this.sub_menus[s_name] }catch(er){alert(er)}
  return null;
}

acWin.prototype.get_main_button_win_obj = function(s_name, width, s_title, button, obj_type)
{
 //alert(JSON.stringify(button));
 var s = 'function MenuBtnWin'+s_name+'(parent)';
 s+='{MenuBtnWin.call(this,parent,my_name_=s_name, my_title=s_title, buttons=button, obj_type, width="width:"+width+"%;");';
 s+='parent.main_menus[this.my_name]=this};';
  //alert(s)
 eval(s);
 s = 'MenuBtnWin'+s_name+'.prototype = Object.create(MenuBtnWin.prototype);';
  //alert(s);
 eval(s);
 s='MenuBtnWin'+s_name;
 return eval(s);
}

acWin.prototype.get_main_button_objs = function()
{
  for (b in this.buttons){
   var s_title=this.buttons[b]["title"]
   var button=this.buttons[b]["sub_buttons"]
   var obj_type=this.buttons[b]["obj_type"]
   var s = 'function MenuBtn'+b+'(parent)';
   s+='{MenuBtn.call(this,parent,my_name_="'+b+'", my_title="'+s_title+'", buttons=button, obj_type="'+obj_type+'", width="width:10%;");';
   s+='parent.main_menus[this.my_name]=this;};';
   eval(s);
   s = 'MenuBtn'+b+'.prototype = Object.create(MenuBtn.prototype);';
   //alert(s)
   eval(s);
   s='try{MenuBtn'+b+'.prototype.create_main_content = '+b+'_create_main_content} catch(er){};';
   //alert(s)
   eval(s);

// NEED TO COMPLETE: it should be similar to the click event of sub buttons
//--------------------------------------------------------------------------
//   s='try{MenuBtn'+b+'.prototype.click = function (event){';
//
//s='SubMenuBtn'+this.my_name+s_name+'.prototype.click = function (event){';
////s+='try{alert("'+this.my_name+s_name+'_click(obj=this, event)");} catch(er){alert("er9026: "+ er)}';
//s+='try{eval("this.parent.parent.onclick"+'+this.parent.id+'+"(tab=this.parent.parent.tab_obj_,win=this.parent.parent, obj=this, event)");} catch(er){alert("err403: "+er)};';
//s+='}';
//eval(s);



   s='new MenuBtn'+b+'(parent=this)';
   eval(s);
  }
}

// -- Pop win --
function PopWin(my_name_, win_name_, win_title_, user_id, atm)
{
  this.atm = atm;
  this.active_popup_win=null;
  //alert(JSON.stringify(this.atm.buttons));
  this.buttons=this.atm.buttons;
  this.tab_nav_links=this.atm.tab_nav_links;
  //--
  this.my_name=my_name_;this.name="win_"+win_name_;
  this.user_id=user_id; var is_scroll_=true;
  this.main_menus = {};this.sub_menus = {};
  acWin.call(this,my_name_=my_name_,win_name=this.name,win_title=win_title_,right="2%",top="30%",is_scroll=is_scroll_,zindex=20,tab_obj_=atm.active_tab,is_nav_panel=true,win_number=0)
}
PopWin.prototype = Object.create(acWin.prototype)
PopWin.prototype.constructor = PopWin;

PopWin.prototype.__init__ = function()
{
  this.set_tab(this.atm.active_tab);
  this.set_title_colors("#fff", "#2196F3");
  //--
  this.set_title(this.win_title_+this.atm.active_tab.tab_name);
  this.atm.active_tab.PopWinObjects[this.my_name]=this;
  this.set_panel();
  this.main_menus["Tab"].btn.click()
}

PopWin.prototype.set_panel = function()
{
  this.main_menu = document.createElement("div");
  this.sub_menu = document.createElement("div");
  this.win_nav_panel.appendChild(this.main_menu);
  this.win_nav_panel.appendChild(this.sub_menu);
  //alert(JSON.stringify(this.buttons));
  this.get_main_button_objs();
}


PopWin.prototype.get_functions_properties_editor = function(tab_,functions_dic,functions_list_dic,dic_properties,settings_list,attributes_list,tab_btn_name="TabContent", properties_func, node_to_delete)
{
  return new FunctionsPropertiesEditor(tab_,functions_dic=functions_dic,functions_list_dic,dic_properties,settings_list,attributes_list,tab_btn_name, properties_func, node_to_delete)
}

// -- MenuBtnWin --
function MenuBtnWin(parent, my_name_, my_title, buttons, obj_type, width="width:10%;")
{
try{
 this.my_name=my_name_;
 this.my_title = my_title;
 this.buttons = buttons;
 this.my_sub_objs={}
 this.btn = document.createElement("button");
 this.parent=parent;
 this.btn.win=parent;
 this.btn.menu_btn_win=this;
 this.btn.setAttribute("id", parent.my_name+"_"+this.my_name);
 this.btn.setAttribute("class", "main_menu_btn");
 this.btn.setAttribute("style", width);
 this.btn.innerHTML = this.my_title;
 this.btn.onclick=function(event){
   this.win.sub_menu.innerHTML="";
   this.win.sub_menus = {};
   try{
      for(b in buttons){
        eval('SubMenuBtn'+this.win.my_name+b+'=this.menu_btn_win.get_sub_button_obj_win(b,buttons[b]["title"],obj_type)');
        eval('this.menu_btn_win.my_sub_objs[b] = new SubMenuBtn'+this.win.my_name+b+'(parent=this.menu_btn_win)');
      }
   if(this.is_run_first_sub_menu){
    this.menu_btn_win.my_sub_objs[b].btn.click()
   }
   } catch(er){alert("er202: "+er)}
   try{
      for(m in this.win.main_menus)
      {this.win.main_menus[m].btn.className=this.win.main_menus[m].btn.className.replace(" active", "")}
      try{this.className+=" active";  } catch(er){}
   } catch(er){alert("er203: "+er)}
 }
 parent.main_menu.appendChild(this.btn);
 } catch(er){alert("Error 101: "+er)}
}


MenuBtnWin.prototype.get_sub_button_obj_win = function(s_name, s_title, obj_type)
{
var s='function SubMenuBtn'+this.my_name+s_name+'(parent)';
s+='{';
s+='var width_=parent.buttons["'+s_name+'"]["width"];';
s+='SubMenuBtn.call(this,parent,my_name_="'+s_name+'",my_title_="'+s_title+'",width="width:"+width_+"%;");';
//s+='alert(parent.my_name+"--1--");alert(parent.parent.name+"--2--");'
s+='parent.parent.sub_menus[s_name]=this;'
s+='};';
eval(s);
s = 'SubMenuBtn'+this.my_name+s_name+'.prototype = Object.create(SubMenuBtn.prototype);';
eval(s);
//s='SubMenuBtn'+this.my_name+s_name+'.prototype.click = '+this.my_name+s_name+'_click;'
s='SubMenuBtn'+this.my_name+s_name+'.prototype.click = function (event){';
//s+='try{alert("'+this.my_name+s_name+'_click(obj=this, event)");} catch(er){alert("er9026: "+ er)}';

s+='try{eval("this.parent.parent.onclick"+'+this.parent.id+'+"(tab=this.parent.parent.tab_obj_,win=this.parent.parent, obj=this, event)");} catch(er){alert("err403: "+er)};';

s+='}';
eval(s);
s='SubMenuBtn'+this.my_name+s_name;
return eval(s);
}


// -- MenuBtn --
function MenuBtn(parent, my_name_, my_title, buttons, obj_type, width="width:10%;")
{
 this.parent=parent;
 this.my_name=my_name_;
 this.my_title = my_title;
 this.buttons = buttons;
 this.my_sub_objs={}
 //alert(JSON.stringify(this.buttons));
 this.btn = document.createElement("button");
 this.btn.parent=this;
 this.btn.setAttribute("id", this.parent.my_name+"_"+this.my_name);
 this.btn.setAttribute("class", "main_menu_btn");
 this.btn.setAttribute("style", width);
 this.btn.innerHTML = this.my_title
 this.btn.onclick=function(event){
   try{this.parent.parent.win_content.innerHTML="";} catch(er){}
   this.parent.parent.sub_menu.innerHTML="";
   this.parent.parent.sub_menus = {};
   // need to remove the alert in the catch part.
   try{
         for(b in this.parent.buttons){
           //alert('SubMenuBtn'+this.parent.my_name+b+'=this.parent.get_sub_button_obj(b,this.parent.buttons[b]["title"],obj_type)');
           eval('SubMenuBtn'+this.parent.my_name+b+'=this.parent.get_sub_button_obj(b,this.parent.buttons[b]["title"],obj_type)');
           //alert('this.parent.my_sub_objs[b] = new SubMenuBtn'+this.parent.my_name+b+'(parent=this.parent)');
           eval('this.parent.my_sub_objs[b] = new SubMenuBtn'+this.parent.my_name+b+'(parent=this.parent)');
         }
   } catch(er){alert("er2021: "+er)}
   try{
         for(m in this.parent.parent.main_menus)
         {this.parent.parent.main_menus[m].btn.className=this.parent.parent.main_menus[m].btn.className.replace(" active", "")}
         try{this.className+=" active";  } catch(er){}
   } catch(er){alert("er2031: "+er)}
   try{
     this.parent.create_main_content();
   } catch(er){alert("er201: "+er)}
 }
 this.parent.main_menu.appendChild(this.btn);
}

MenuBtn.prototype.get_sub_button_obj = function(s_name, s_title, obj_type)
{
 var s='function SubMenuBtn'+this.my_name+s_name+'(parent)';
 s+='{';
 s+='var width_=parent.buttons["'+s_name+'"]["width"];';
 s+='SubMenuBtn.call(this,parent,my_name_="'+s_name+'",my_title_="'+s_title+'",width="width:"+width_+"%;");';
 s+='parent.parent.sub_menus[this.my_name]=this;};';
 eval(s);
 s = 'SubMenuBtn'+this.my_name+s_name+'.prototype = Object.create(SubMenuBtn.prototype);';
 eval(s);
 //s='SubMenuBtn'+this.my_name+s_name+'.prototype.click = '+this.my_name+s_name+'_click;'
 s='SubMenuBtn'+this.my_name+s_name+'.prototype.click = function (event){';
 s+='this.obj_name="ac'+ s_name+'";';
 if(obj_type!="none"){
   s+='this.parent.parent.atm.active_tab.set_to_create_obj(dic={"parent_obj_name":"'+this.my_name+'", "obj_type":"'+obj_type+'", "obj_name":"ac'+s_name+'", "element_name":"'+s_name+'",'
   s+='"properties":{"title":"'+s_title+'"}, "functions":{}, "attributes":{}'
   s+='});'
 }
 s+='try{'+this.my_name+s_name+'_click(obj=this, event);} catch(er){}';
 s+='}';
 //alert(s);
 eval(s);
 s='SubMenuBtn'+this.my_name+s_name;
 return eval(s);
}

// -- SubMenuBtn --
function SubMenuBtn(parent, my_name_, my_title_, width="width:10%;")
{
 this.parent=parent;this.my_title=my_title_;this.my_name_=my_name_;this.my_name=parent.my_name+my_name_;
 this.btn = document.createElement("button");
 this.btn.parent=this;
 this.btn.setAttribute("id", this.my_name);
 this.btn.setAttribute("class", "main_menu_btn");
 this.btn.setAttribute("style", width);
 this.btn.innerHTML = my_title_;
 this.btn.onclick=function(event){
   //alert(this.parent.click)
   this.parent.click(event)
   }
 parent.parent.sub_menu.appendChild(this.btn);
}

// -- MenuBtnTab --
Tab_create_main_content = function()
{
  //alert("1 Tab_create_main_content: " + this.parent.atm.active_tab.tab_name)

//
//    this.parent.nav=document.createElement("div");
//    this.parent.nav.onclick=function(){
//      //alert(JSON.stringify(editor_.tab_obj_.tab_functions))
//      var e=event.target;
//      try{var cc=e.getAttribute("class");if(cc!="funtablinks"){return;}} catch(er){}
//      var f=event.target.innerHTML;
//      editor.tab_obj_.active_function=f;
//      editor.tab_content.innerHTML=editor.tab_obj_.tab_functions[f];
//      editor.tab_content.setAttribute("fun_name",f);
//      var funtablinks = document.getElementsByClassName("funtablinks");
//      for (var i=0;i<funtablinks.length;i++){funtablinks[i].className=funtablinks[i].className.replace(" active","");};
//      event.target.className += " active";
//       //alert(editor_.tab_content.outerHTML)
//    }.bind(editor=this.parent, event);
//    this.parent.nav.setAttribute("class", "tab");
//    this.parent.nav.btns={};
//    this.parent.win_content.appendChild(this.parent.nav);
//    this.parent.tab_content = document.createElement("textarea");
//    this.parent.tab_content.setAttribute("class", "tab_textarea");
//    this.parent.tab_content.onchange= function (){
//      editor.tab_obj_.tab_functions[event.target.getAttribute("fun_name")]=event.target.value;
//      editor.tab_obj_.parent.save();
//      var click_event = new Event("click", {bubbles: true});
//      //editor.main_menus["Tab"].btn.dispatchEvent(click_event);
//    }.bind(editor=this.parent, event)
//    this.parent.win_content.appendChild(this.parent.tab_content);
//
//    var tfs_ = Object.keys(this.parent.tab_obj_.tab_functions)
//    var fs_=[]; var fs__ = this.parent.tab_obj_.parent.tab["functions"]; for(i in fs__){fs_.push(fs__[i])}
//    for(i in tfs_){var f = tfs_[i];if(!(f in fs_)){fs_.push(f)}}
//    for(i in fs_)
//    {
//      var f = fs_[i];
//      this.parent.nav.btns[f] = document.createElement("button");
//      this.parent.nav.btns[f].setAttribute("class", "funtablinks");
//      this.parent.nav.btns[f].innerHTML = f;
//      this.parent.nav.appendChild(this.parent.nav.btns[f]);
//    }
//
//  // -- need to complete
//  this.parent.tab_properties_ = document.createElement("div");
//  var table = document.createElement("table");var tr=document.createElement("tr");table.appendChild(tr);
//  var thp=document.createElement("th");thp.innerHTML="Property";thp.setAttribute("style","width:10%;text-align:center;");
//  tr.appendChild(thp);
//  var thv=document.createElement("th");thv.innerHTML="Value";thv.setAttribute("style","width:10%;text-align:center;");
//  tr.appendChild(thv);
//  this.parent.tab_properties_.appendChild(table);
//
//  // alert(this.parent.tab_obj_.tab_name);
//  var ps_=[];var ps__=this.parent.tab_obj_.parent.tab["settings_list"];for(i in ps__){ps_.push(ps__[i])}
//  var tps=this.parent.tab_obj_.tab_properties;var tps_=Object.keys(tps);for(i in tps_){var p=tps_[i];if(!(p in ps_)){ps_.push(p)}}
//  for(i in ps_)
//  {
//   var s=ps_[i];
//   var tr=document.createElement("tr");table.appendChild(tr);
//   var td=document.createElement("td");td.innerHTML=s;tr.appendChild(td);
//   var td=document.createElement("td");var input=document.createElement("input");
//   input.setAttribute("property",s);td.appendChild(input);
//   var s_=tps[s];try{if(s_==null){}else{input.value=s_}} catch(er){alert("er9027: "+ er)};tr.appendChild(td);
//  }
//
//  this.parent.tab_properties_.addEventListener("change", function(){
//    var p=event.target;var property=p.getAttribute("property");var v=p.value;
//    editor.tab_obj_.tab_properties[property]=v;
//    //--
//    editor.tab_obj_.parent.save();
//    //--
//    // alert(JSON.stringify(editor.tab_obj_.tab_properties))
//  }.bind(editor=this.parent, event))
//  this.parent.tab_properties_.setAttribute("class", "com_setting");
//  this.parent.win_content.appendChild(this.parent.tab_properties_);
}


// -- MenuBtnComponent --
Component_create_main_content = function()
{
  //alert(this.my_name)
  //--  alert(editor.my_name)
  this.parent.component_left_nav = document.createElement("div");
  this.parent.component_left_nav.setAttribute("class", "tab");
  this.parent.component_left_nav.btns={};
  this.parent.component_left_nav.onclick=function(){
    var f=event.target.innerHTML;
    editor.tab_obj_.active_component_function=f;
    if(editor.tab_obj_.active_obj.data["functions"][f]==null)
    {editor.component_fun_editor.innerHTML="function (event){\ntry{\n\n} catch(er){alert('er9026: '+ er)}\n}";} else
    {editor.component_fun_editor.innerHTML=editor.tab_obj_.active_obj.data["functions"][f]}

    var comfuntablinks = document.getElementsByClassName("comfuntablinks");
    for (var i=0;i<comfuntablinks.length;i++){comfuntablinks[i].className=comfuntablinks[i].className.replace(" active","");};
    event.target.className += " active";
  }.bind(editor=this.parent, event);
  this.parent.win_content.appendChild(this.parent.component_left_nav);
  //--
  this.parent.component_fun_editor = document.createElement("textarea");
  this.parent.component_fun_editor.setAttribute("class", "editor");
  this.parent.component_fun_editor.addEventListener("change", function(){
    var f=event.target.value;
    editor.tab_obj_.active_obj.data["functions"][editor.tab_obj_.active_component_function]=f;
    var obj_number = editor.tab_obj_.active_obj.data["properties"]["obj_number"];
    editor.tab_obj_.tab_objects[obj_number] = editor.tab_obj_.active_obj.data;
    editor.tab_obj_.parent.save();
  }.bind(editor=this.parent, event))
  this.parent.win_content.appendChild(this.parent.component_fun_editor);
  // --
  this.parent.component_properties = document.createElement("div");
  this.parent.component_properties.addEventListener("change", function(){
    var p=event.target; var property=p.getAttribute("property");var v=p.value;
    editor.tab_obj_.active_obj.data["properties"][property]=v;
    //--
    var obj_number = editor.tab_obj_.active_obj.data["properties"]["obj_number"];
    editor.tab_obj_.tab_objects[obj_number] = editor.tab_obj_.active_obj.data;
    editor.tab_obj_.parent.save();
  // --
    //alert(JSON.stringify(editor.tab_obj_.tab_content))
  }.bind(editor=this.parent, event))
  this.parent.component_properties.setAttribute("class", "com_setting");
  this.parent.win_content.appendChild(this.parent.component_properties);
  // --
}


ComponentButton_click = function(obj, event)
{
  //alert(2223)
  //alert(obj.parent.parent.tab_obj_.tab_name)
  //alert(obj.parent.parent.my_name); // editor
  //alert(obj.parent.parent.atm.my_name); // atm
  //alert(obj.parent.my_name); // Component

  //alert(obj.obj_name)
}

ComponentSpan_click = function(obj, event)
{
  //alert(obj.parent.parent.tab_obj_.tab_name)
  //alert(obj.parent.parent.my_name)
  //alert(obj.parent.my_name)
  //alert(obj.obj_name)
}

ComponentInput_click = function(obj, event)
{
  //alert(obj.parent.parent.tab_obj_.tab_name)
  //alert(obj.parent.parent.my_name)
  //alert(obj.parent.my_name)
  //alert(obj.obj_name)
}


//-- TabNavLink --
TabNavLink_create_main_content = function()
{
 //alert("TabNavLink_create_main_content")
 //this.my_sub_objs["nav"].click()
}

//- nav --
TabNavLinknav_click = function(obj, event)
{
//  alert(obj.my_name);
//  alert(obj.parent.my_name) // TabNavLink
//  alert(obj.parent.parent.my_name) // editor
//  alert(obj.parent.parent.atm.active_tab.tab_name)
  //alert(JSON.stringify(obj.parent.parent.tab_nav_links))
  //alert(JSON.stringify(obj.parent.parent.atm.tab_nav_links))
  //alert(JSON.stringify(obj.parent.parent.atm.active_tab.tab_nav_links))

  try{tab_fpe=obj.parent.parent.get_functions_properties_editor(
            obj.parent.parent.atm.active_tab,
            functions_dic=obj.parent.parent.atm.active_tab.tab_nav_links["functions"],
            functions_list_dic=obj.parent.parent.atm.tab_nav_links["functions"],
            dic_properties=obj.parent.parent.atm.active_tab.tab_nav_links["properties"],
            settings_list=obj.parent.parent.atm.tab_nav_links["settings_list"],
            attributes_list=obj.parent.parent.atm.tab_nav_links["attributes_list"],
            tab_btn_name="TabNavLink",null,
            node_to_delete=".tab_nav_links")} catch(er) {alert("er9030: "+ er)}
}

//- item --
TabNavLinkitem_click = function(obj, event)
{
  //alert("TabNavLink- item")
  //alert(obj.obj_name) // acitem
//  alert(obj.parent.my_name) // TabNavLink (in editor. it is the MenuBtn)
//  alert(obj.parent.parent.my_name) // editor
  //alert(obj.parent.parent.atm.active_tab.tab_name)
  var link_number=obj.parent.parent.active_link.link_number
  //alert(link_number)
  //alert(JSON.stringify(obj.parent.parent.atm.active_tab.tab_nav_links["nav_links"][link_number]))
  //alert(JSON.stringify(obj.parent.parent.atm.tab_nav_links))
  try{tab_fpe=obj.parent.parent.get_functions_properties_editor(
            obj.parent.parent.atm.active_tab,
              functions_dic=obj.parent.parent.atm.active_tab.tab_nav_links["nav_links"][link_number]["nav_link"]["functions"],
            functions_list_dic=obj.parent.parent.atm.nav_link["functions"],
              dic_properties=obj.parent.parent.atm.active_tab.tab_nav_links["nav_links"][link_number]["nav_link"]["properties"],
            settings_list=obj.parent.parent.atm.nav_link["settings_list"],
            attributes_list=obj.parent.parent.atm.nav_link["attributes_list"],
            tab_btn_name="TabNavLink",null,
            node_to_delete='.tab_nav_links["nav_links"]['+link_number+']')} catch(er) {//alert("er9031: "+ er)
      }
}


// -- PopWin create_main_content--
PopWin_create_main_content = function()
{
 //alert(11111)
}

PopWinNewPopWin_click = function(obj, event)
{
}

PopWinDeletePopWin_click = function(obj, event)
{
//  alert("PopWinDeletePopWin")
//  alert(obj.obj_name) // acDeletePopWin
//  alert(obj.parent.my_name) // PopWin (in editor. it is the MenuBtn)
//  alert(obj.parent.parent.my_name) // editor
//  alert(obj.parent.parent.tab_obj_.tab_name)
//  alert(obj.parent.parent.tab_obj_.tab_name)
//alert(JSON.stringify(obj.parent.parent.tab_obj_.tab_pop_win_buttons))

//var ll=obj.parent.parent.active_popup_win;
//delete obj.parent.parent.tab_obj_.tab_pop_win_buttons["pop_wins"][ll[2]]
//this.tab_obj_.parent.save();
}


//-- TabContent --
TabContent_create_main_content = function()
{
 //alert(22212000)
}

TabContentcontent_click = function(obj, event)
{
  //alert(obj.btn)
  //alert(obj.btn.outerHTML);
  //alert(obj.my_name) // TabContentcontent
  //alert(obj.parent.my_name) // TabContent
  //alert(obj.parent.parent.my_name) // editor
//  alert(obj.parent.parent.atm.active_tab.tab_name)
//  alert(131)
//  alert(JSON.stringify(obj.parent.parent.tab_nav_links))
//  alert(JSON.stringify(obj.parent.parent.atm.tab_nav_links))
//  alert(JSON.stringify(obj.parent.parent.atm.active_tab.tab_nav_links))
//   alert(133)
  //var tab_=obj.parent.parent.atm.active_tab;

try{
  var my_creator_obj_=obj.parent.parent.atm.active_tab_content.my_creator_obj;
} catch(er){alert(er)}

try{
  var t_=""; t_=obj.parent.parent.atm.active_tab_content.getAttribute("my_type");
} catch(er){alert(er)}

  if(my_creator_obj_.is_link){
     var tab_content_=obj.parent.parent.atm.active_tab.tab_nav_links["nav_links"][my_creator_obj_.link_number]["tab_content"]
  } else if (t_=="TabPopWin")
  {var tab_content_=my_creator_obj_.link_dic
  } else {var tab_content_=obj.parent.parent.atm.active_tab.tab_content_link_dic;}

  var tab_content=obj.parent.parent.atm.tab_content

  try{
       //alert("787878787878-11 "+JSON.stringify(tab_content_))
      tab_fpe=obj.parent.parent.get_functions_properties_editor(
            obj.parent.parent.atm.active_tab,
            functions_dic=tab_content_["functions"],
            functions_list_dic=tab_content["functions_list"],
            dic_properties=tab_content_["properties"],
            settings_list=tab_content["settings_list"],
            attributes_list=tab_content["attributes_list"],
            tab_btn_name="TabContent",
            null,
            node_to_delete=null
        )
  } catch(er) {alert('er9036: '+ er)}
}


//-- Plugin --
Admin_create_main_content = function()
{
 //alert(333)
}

//-- Plugin --
Plugin_create_main_content = function()
{
 //alert(333)
}

PluginSearchTable_click = function(obj, event)
{
 //alert(333444)
  //var tab_=obj.parent.parent.atm.active_tab;
}

PluginMSearchTable_click = function(obj, event)
{
 //alert(333444)
  //var tab_=obj.parent.parent.atm.active_tab;
}

//-- Report --
Report_create_main_content = function()
{
 //alert("333 Report")
}

ReportPivot_click = function(obj, event)
{
 //alert(333444)
}

// --
Dashboard_create_main_content = function()
{
 //alert("4444 Dashboard_create_main_content")
}

ReportBasic_click = function(obj, event)
{
 alert("55555 ReportBasic_click")
}

//-- AppObjs --
AppObjs_create_main_content = function()
{
 // alert(this.parent.my_name); // it is the editor
 var f=function(select_){select_.innerHTML="";var o=document.createElement("option");o.value="";o.innerHTML="------";select_.appendChild(o);
 for(var j in atm.app_content){var o=document.createElement("option");o.value=j;o.innerHTML=atm.app_content[j]["properties"]["obj_name"];select_.appendChild(o)}
 }
 var btn=document.createElement("button");btn.setAttribute("class", "main_menu_btn");btn.setAttribute("style", "width:150px");btn.innerHTML="create new AppObj";
 var span_=document.createElement("span");span_.setAttribute("style","width:150px;text-align:rigth");span_.innerHTML="  Edit AppObj: ";
 var select=document.createElement("select");select.setAttribute("style","width:150px");f(select);
 select.addEventListener("change", function(){
  var e=event.target;
  try{var tab_fpe_=atm.editor.get_functions_properties_editor(
              atm.active_tab,functions_dic=atm.app_content[e.value]["functions"],functions_list_dic={},
              dic_properties=atm.app_content[e.value]["properties"],settings_list={}, attributes_list={"obj_name":[]},
              tab_btn_name="AppObjs",properties_func=null,node_to_delete=".parent.app_content["+e.value+"]")
    } catch(er) {}
 })

 btn.onclick=function(event){
   //alert(atm.my_name)
   //alert(JSON.stringify(atm.app_content))
   var app_obj_name_ = prompt("Enter name for new AppObj:",'');if(app_obj_name_==null || app_obj_name_==''){alert("Please enter a name for AppObj"); return;}
   var n_=atm.get_next_obj_number();
   atm.app_content[n_]={"functions":{"constructor": "function (obj){try{\nthis.obj=obj;\n\n}catch(er){'error 5000'}}"},
                        "properties":{"obj_name":app_obj_name_},
                        "settings":{}}
   atm.save(); f(this.select);
 };
 btn.select=select;
 this.parent.win_content.appendChild(btn);this.parent.win_content.appendChild(span_);this.parent.win_content.appendChild(select);
}

AppObjsCreateObject_click = function(obj, event)
{
 alert("222 create AppObjs")
}


//-- General Functions --
var getEBI=function(s){s=s.toString();return document.getElementById(s)}
var get_creator=function(n){return getEBI(n).my_creator_obj;}
var clone_dic=function(dic){return JSON.parse(JSON.stringify(dic))}
//
var activate_function_of_obj=function(obj_name,fun_name,params="")
{var o=eval("atm."+obj_name);return eval("o."+fun_name+"("+params+")")}

var get_value_from_dic=function(dic,from_list,value,to_list)
{for (var k in dic[from_list]){if(dic[from_list][k]==value){return dic[to_list][k]}}}

var get_date_idx=function(type="basic"){var date = new Date;
 var r = date.getFullYear()+complete_zeros((date.getMonth()+1)+"",2)+complete_zeros(date.getDate()+"",2)+
 complete_zeros(date.getHours()+"",2)+complete_zeros(date.getMinutes()+"",2);
 if(type=="s"){r={"ymdht":r, "s":date.getSeconds(), "h":date.getHours(), "t":date.getMinutes()}}
 return r
}

var nice_number = function(z){
 var kk="";if(z*1<0){kk="-";z=-1*z};
 var s=z+"";s_=s.split(".");
 var ns=s_[0];var dn=s_[1];
 var ns_=0;
 if(dn){if(dn.length<2){dn=dn+"0"} else if(dn.length>2)
 {dn=Math.round((100*dn)/Math.pow(10, dn.length))/100;if(dn==1 || dn==0){ns_=1*dn;dn="00"} else{dn=dn+"";dn=dn.split(".")[1]}}}
 else{dn="00"}

 var nss=ns.split(",");var n_="";for(var j=0;j<nss.length;j++){n_= n_+nss[j]};var n__="";n_=1*n_+ns_;n_=""+n_;
 while (n_.length>0) {if(n__.length>0){n__=","+n__};n__=n_.substring(n_.length-3,n_.length)+n__;n_=n_.substring(0,n_.length-3)}
 return kk+n__+"."+dn;
}
var complete_zeros = function(s,n){for(var i=0;i<(n-s.length);i++){s="0"+s};return s;}

var clean_number = function(z){
 try{if(z==null){return z}; z=""+z;z=z.replace(/,/g, ''); z=1*z}catch(er){}
 return z
}
var num_to_day = function(n){return days_of_name[n]}
//-- General Dictionaries --
var months_dic={"01":"Jan", "02":"Feb","03":"Mar", "04":"Apr","05":"May", "06":"Jun",
                "07":"Jul", "08":"Aug","09":"Sep", "10":"Oct","11":"Nov", "12":"Dec"}
var days_of_name=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
//-- Other Functions --

var list_average=function(l){return Math.round(100*l.reduce((a,b) => a+b, 0)/l.length)/100;}

const list_variance = (arr = []) => {if(!arr.length){return 0};
   const sum = arr.reduce((acc, val) => acc + val); const { length: num } = arr; const median = sum / num;
   let variance = 0; arr.forEach(num => {variance += ((num - median) * (num - median));});variance /= num;
   return Math.round(100*variance)/100;
};

const list_std = function(arr = []){return Math.round(100*Math.sqrt(list_variance(arr)))/100};

// Binomial --
const binomial=function(n,p,k){var c=Math.pow(p, k)*Math.pow(1-p, n-k);
//alert("k="+k + "  n="+n +"   p="+p +"   c="+c)
if(n>180 && ((k > 150) )){
  var nn=10;var h=Math.round(k/nn)
  for(var x=n-h+1;x<=n;x++){c=c*x;};
    //alert("A cc1="+k+" "+c)
  for(var z=1;z<h;z++){c=c/z;};
    //alert("cc2="+c)
  for(var g=1;g<nn-1;g++){for(var x=n-(g+1)*h+1;x<n-g*h+1;x++){c=c*x;}; for(var z=g*h;z<(g+1)*h;z++){c=c/z;};}
    //alert("ccccc = "+c)
  for(var x=n-k+1;x<n-(nn-1)*h+1;x++){c=c*x;};
    //alert("cc12="+c)
  for(var z=(nn-1)*h;z<=k;z++){c=c/z;};
    //alert("c3="+c)
} else{
    for(var x=n-k+1;x<=n;x++){c=c*x;};for(var z=1;z<=k;z++){c=c/z;};
}
return c;}

//var s=0;
//var n=1000
// var l=[]
// for(var i=0;i<=1000;i++)
// {var b=binomial(n,0.50068,i);s+=b;l.push(b)}
//
//alert(n+"\n\n"+s+"\n\n"+l)




