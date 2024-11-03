// -- SimTab object --
function SimTab(parent, name, title, game_id, url_get_screens)
{
 this.parent = parent;
 this.name = name;
 this.title = title;
 this.need_update_screen = 1;
 this.NAV = null;
 this.CONTENT = null;
 this.CONTENTObjects = {};
 this.GAME_ID = game_id;
 this.is_show_first = true;
 //--
  ed = document.createElement("div")
  ed.setAttribute("class", "tab-pane fade "+this.parent.sss)
  ed.setAttribute("id", this.name)
  // alert(ed.outerHTML)
  this.parent.tabContents.appendChild(ed)
  //
  e = document.createElement("li")
  e.setAttribute("class", "nav-item")
  ea = document.createElement("a")
  ea.setAttribute("class", "nav-link "+this.parent.ss)
  ea.setAttribute("id", "tab-"+name)
  ea.setAttribute("obj_name", name)
  ea.setAttribute("href", "#"+name)
  ea.innerHTML = title
  e.appendChild(ea)
  this.parent.tabTitles.insertBefore(e, this.parent.tabTitles.childNodes[0])
  ea.addEventListener("click", function(){
    event.preventDefault();
    elm = event.target; obj_name = elm.getAttribute('obj_name');
    obj = parent.TabObjects[obj_name]
    is_show_first = obj.is_show_first
      obj.setScreens(game_id, url_get_screens, is_show_first);
      $(this).tab('show');
      post_ajax_create_action('globsim__change_to_tab__' + obj.title)
  }.bind(ea, event, parent ));
 this.tabTitle = ea; // document.getElementById("tab-"+this.name)
 this.tabDoc = ed; //document.getElementById(this.name)
}

SimTab.prototype.setScreens = function(game_id, url_get_screens, is_show_first=false) {
   // alert('SimTab.prototype.setScreens')
   console.log('SimTab.prototype.setScreens', game_id)
   doc_ = this.tabDoc
   sim_tab = this;
   if (this.need_update_screen == 1)
   {
      $.post(url_get_screens,
        {
          name: this.name,
          obj_id: game_id
        },
        function(data){
            doc_.innerHTML = data
            try{
            sim_tab.set_nav_prefix()
            } catch (e) {alert(e)}
        });
      this.need_update_screen = 0;
      //log(this.name + "--setScreens, is_show_first= "+is_show_first)
      if(is_show_first){folder = this.name.toLowerCase(); setTimeout(set_first_obj, 1, folder)}
   }
}

SimTab.prototype.set_nav_prefix = function() {
    obj_name = this.name.toLowerCase()+"_nav"
    var obj = document.getElementById(obj_name)
    if (!obj) {setTimeout(this.set_nav_prefix, 1); return;}
    else {this.NAV_PREFIX =  obj.getAttribute("nav_prefix") }
}

set_first_obj = function(folder, first=null, obj_name='')
{
    var projects = document.getElementById(folder+"_projects_")
    if (!projects) {setTimeout(set_first_obj, 1, folder, first, obj_name); return;}
    else{
       if (obj_name != '')
       {
         var obj = document.getElementById(obj_name)
         if (!obj) {setTimeout(set_first_obj, 1, folder, first, obj_name); return;}
         else {
          first = obj;
          var event = new Event("click", {bubbles: true});
          first.dispatchEvent(event);
         }
       } else {
        if (!first){first = projects.children[0].children[1]}
         var event = new Event("click", {bubbles: true});
         first.dispatchEvent(event);
       }
    }
}

SimTab.prototype.click_main = function(obj_id)
{
    sim_tab = this;
    name_ = this.name.toLowerCase(); content_ = name_ + "_content"; nav_ = name_ + "_nav"
    this.NAV = document.getElementById(nav_)
    this.CONTENT = document.getElementById(content_)
    contents_ = this.CONTENTObjects
    for (d in contents_)
    {dd = contents_[d];dd.style.display = 'none';this.get_nav(d).style.color = "black";}
    if (obj_id != "new")
    {
      this.run_code_insure_obj(obj_id, "document.getElementById('"+this.NAV_PREFIX+"_"+obj_id+"').style.color = 'blue'");
      post_ajax_create_action('globsim__change_to_item__' + name_ + '_' + obj_id)
      doc_ = contents_[obj_id]; if(doc_){doc_.style.display = 'block'; return -1;}
    }
    var obj_name = ""
    if (obj_id == "new")
    {obj_name = prompt('Enter R&D Project name:', '');
     if(obj_name == "" || obj_name == null) {return -1;}
    }
      doc_ = this.CONTENT
      $.post(link3,
        {
          app: "globsim",
          model_name: this.name,
          obj_id: obj_id,
          name: obj_name,
          game_id: this.GAME_ID
        },
        function(data){
            div_ = document.createElement("div")
            div_.innerHTML = data
            table_ = div_.getElementsByTagName("table")
            obj_id_ = table_[0].getAttribute('pkey')
            div_.setAttribute('id', name_ + '_obj_' + obj_id_)
            div_.setAttribute('pkey', obj_id_)
            doc_.appendChild(div_)
            contents_[obj_id_] = div_

            if (obj_id == "new")
            { try{sim_tab.callback_new(div_, obj_id_)} catch (er) {alert(er)} }
            s_obj = "document.getElementById('" + sim_tab.NAV_PREFIX + "_"+ obj_id_ + "').style.color = 'blue'"
            sim_tab.run_code_insure_obj(obj_id_, s_obj)
            if (name_=="randd")
            {
                t = document.getElementById("rd__attributes_"+obj_id_)
                inputs_ = t.getElementsByTagName("input")
                for (i in inputs_)
                {
                    if(inputs_[i].className)
                    {
                     if(inputs_[i].className.indexOf('slider') >= 0)
                     {var event = new Event("change", {bubbles: true});inputs_[i].dispatchEvent(event);}
                    }
                }
            }
        });
}

SimTab.prototype.get_nav = function(obj_id) {
    return document.getElementById(this.NAV_PREFIX + "_" + obj_id)
}

SimTab.prototype.run_code_insure_obj = function(obj_id, run_code="") {
    obj = document.getElementById(this.NAV_PREFIX + "_" + obj_id)
    if (!obj) {setTimeout(this.run_code_insure_obj, 1, obj_id, run_code); log(obj_id); return}
    else { try{eval(run_code)} catch (er) {alert(er)}}
}


// =========== Management ==========================
// -- SimManagementTab object --
function SimManagementTab(parent, name, game_id, url_get_screens)
{
 this.parent = parent;
 this.name = name;
 this.NAV = null;
 this.CONTENT = null;
  //--
  ed = document.getElementById(this.name)
  //this.parent.tabContents.appendChild(ed)
  ea = document.getElementById("tab-"+this.name)
  //e = ea.parentNode
  //this.parent.tabTitles.appendChild(e)
  //
  ea.addEventListener("click", function(){
    event.preventDefault();
    elm = event.target; obj_name = elm.getAttribute('obj_name');
    obj = parent.TabObjects[obj_name]
      obj.setScreens(game_id, url_get_screens);
      $(this).tab('show');
  }.bind(ea, event, parent));
 this.tabTitle = ea; // document.getElementById("tab-"+this.name)
 this.tabDoc = ed; //document.getElementById(this.name)
//
  name_ = this.name.toLowerCase(); content_ = name_ + "_content"; nav_ = name_ + "_nav"
  this.NAV = document.getElementById(nav_)
  this.CONTENT = document.getElementById(content_)
}

SimManagementTab.prototype.setScreens = function(game_id, url_get_screens){
   doc_ = this.tabDoc
}

// -- SimManager --
function SimManager(game_id, url_get_screens)
{
    this.game_id = game_id;
    this.url_get_screens = url_get_screens
    this.name = "SimGlob"
    this.tabTitles = document.getElementById("tab_titles")
    this.tabContents = document.getElementById("tab_contents")
    this.TabObjects = {}
}

SimManager.prototype.setTabs = function(tabs, tabs_management, is_staff='False') {
    for(var name in tabs)
    {
      if(name=="RandD"){this.ss = "active"; this.sss="show active"} else {this.ss=""; this.sss=""}
      s = "this.TabObjects[name] = new SimTab(this, name, tabs[name], this.game_id, this.url_get_screens)"
      eval(s)
    }
    if (is_staff == 'True'){
        for(var name in tabs_management)
        {
          s = "this.TabObjects[name] = new SimManagementTab(this, name, this.game_id, this.url_get_screens)"
          eval(s)
        }
    }
    this.TabObjects["RandD"].setScreens(this.game_id, this.url_get_screens)
}

// == Math ==
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}
