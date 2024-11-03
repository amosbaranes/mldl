// base win object
//----------------
function acWin(my_name_="none", win_name="none", win_title="none", right= "0%", top="0%",
  is_scroll=true, zindex="11", tab_obj_=null, is_nav_panel=false)
{
  // create its div for window
  //alert(win_name +" : " + win_title +" this.width_: " + this.width_ +" : " + height +" : " + right +" top: " + top)
  this.is_nav_panel = is_nav_panel;
  this.nav_height = 0;
  this.is_scroll = is_scroll
  this.win_name = win_name
  this.my_name_ = my_name_
  this.tab_obj_ = tab_obj_
  this.win_frame = document.createElement("div");
  this.win_frame.setAttribute("id", "win_frame_"+this.win_name);
  this.win_frame.setAttribute("my_name", my_name_);
  // create its title for window
  this.title_height = 25
  this.win_frame_title = document.createElement("div");
  this.win_frame_title.setAttribute("id", "win_frame_title_"+this.win_name);
  // --
  this.set_title_colors("#fff", "#2196F3");
  // --
  this.win_frame_ico = document.createElement("img");
  this.win_frame_ico.setAttribute("id", "win_frame_ico_"+this.win_name);
  this.win_frame_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_frame_ico.setAttribute("style", "border-radius: 10px;position: absolute;left: 45px");
  this.win_frame_ico.setAttribute("height", "20");
  this.win_frame_ico.setAttribute("width", "20");
  this.win_frame_title.appendChild(this.win_frame_ico)
  this.win_frame_title_span = document.createElement("span");
  this.win_frame_title_span.innerHTML = win_title
  this.win_frame_title.appendChild(this.win_frame_title_span)

  this.win_frame_msg_span = document.createElement("span");
  this.win_frame_title.appendChild(this.win_frame_msg_span);

  this.win_frame.appendChild(this.win_frame_title)
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
  this.win_content = document.createElement("div");
  this.win_content.setAttribute("id", "win_content_"+this.win_name);
  // --
  this.set_win_frame_style(zindex, height="300px", width="300px", right, top, "white");
  //--
  this.win_frame.appendChild(this.win_content)
  //--
  this.win_nav = document.createElement("div");
  this.win_nav.setAttribute("id", "win_nav_"+this.win_name);
  this.style_win_nav = "position:fixed;text-align:center;width:10px;display:none; right:"+right+";top:"+top+";";
  this.win_nav.setAttribute("style", this.style_win_nav)
  // --
  this.win_nav_ico = document.createElement("img");
  this.win_nav_ico.setAttribute("id", "win_nav_ico_"+this.win_name);
  this.win_nav_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_nav_ico.setAttribute("style", "border-radius: 10px;position: absolute;right:97%");
  this.win_nav_ico.setAttribute("height", "20");
  this.win_nav_ico.setAttribute("width", "20");
  this.win_nav.appendChild(this.win_nav_ico)
  // --
  this.span = document.createElement("span");
  this.span.setAttribute("id", "span_"+this.win_name);
  //this.span.innerHTML = "&nbsp;Whiteboard"
  this.span.setAttribute("style", "position: absolute;left: 50");
  this.win_nav.appendChild(this.span)
  // --
  var body_ = document.getElementById("body_base0")
  body_.appendChild(this.win_frame)
  body_.appendChild(this.win_nav)
  // --
  this.win_frame_title.setAttribute('pos1', 0);
  this.win_frame_title.setAttribute('pos2', 0);
  this.win_frame_title.setAttribute('pos3', 0);
  this.win_frame_title.setAttribute('pos4', 0);
  this.win_frame.style.display = "none"
  this.win_frame_style_display = this.win_frame.style.display;
}

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
  this.win_frame.setAttribute("style", this.style_frame);
  this.style_content = "height:"+(height-this.title_height-7-this.nav_height)+"px;width:"+width+"px;"
  if(this.is_scroll==true){this.style_content += "overflow: scroll;"}
  this.win_content.setAttribute("style", this.style_content);
}

acWin.prototype.close_win = function()
{this.win_frame.style.display = "none";}

acWin.prototype.resume_win = function()
{this.win_frame.style.display = this.win_frame_style_display;}

acWin.prototype.set_acWinStat = function(ss)
{
    //open and close acWin.
    if (ss != this.win_frame.style.display)
    {
      var event_close_ac = new Event("click", {bubbles: true});
      this.win_frame_ico.dispatchEvent(event_close_ac);
      this.win_frame_style_display = ss;
    }
    this.tab_obj_.set_max_zindex(this.win_frame);
}

acWin.prototype.set_title = function(title)
{
 this.win_frame_title_span.innerHTML = title;
 this.set_acWinStat('block');
}

acWin.prototype.set_msg = function(msg)
{
 this.win_frame_msg_span.innerHTML = msg;
 this.set_acWinStat('block');
}

acWin.prototype.get_msg = function()
{
 return this.win_frame_msg_span.innerHTML;
}

acWin.prototype.set_info_by_ticker = function(content, win_title)
{
 this.win_content.innerHTML = content;
 this.win_frame_title_span.innerHTML = win_title;
 this.set_acWinStat('block');
}

acWin.prototype.add_info_by_ticker = function(content, win_title)
{
 this.win_content.innerHTML += content;
 this.win_frame_title_span.innerHTML = win_title;
 this.set_acWinStat('block');
}

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
  eval("var " +  ss_obj.my_name + '= ss_obj')
  //console.log("var " +  ss_obj.my_name + '= ss_obj');
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
   console.log("eval 100:"+ss_obj, s)
   eval(s)
   console.log("Good :set_acWinStatEventListeners: ", "eval 100")
   } catch (err) {console.log("Error 101", err.message)}

  console.log("set_acWinStatEventListeners :", "Check 0100");
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

  console.log("set_acWinStatEventListeners :", "Check 0200");
  s = ''
  s += 'this.win_frame_title.addEventListener("mousedown", function(){'
  s += 'event.preventDefault();'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos3", event.clientX);'
  s += 'elm_win_frame_title_'+this.win_name+'.setAttribute("pos4", event.clientY);'

  s += 'document.addEventListener("mouseup", my_mouse_up_'+this.win_name+' = function(){'
  s += 'event.preventDefault();'

  s += 'try{'
  s += 'document.removeEventListener("mouseup", my_mouse_up_'+this.win_name+');'
  s += 'document.removeEventListener("mousemove", my_mouse_move_'+this.win_name+');'
  s += '} catch (err) {err.message}'

  // s += 'alert(elm_win_frame_' +this.win_name +'.outerHTML);'
  s += 'tab_obj_.set_max_zindex(elm_win_frame_' +this.win_name +');'

  s += '}.bind(elm_win_frame_title_'+this.win_name+', event, tab_obj_, elm_win_frame_' +this.win_name + '));'

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
  s += ' = this.win_frame, elm_win_nav_'+this.win_name+' = this.win_nav));'
  try{
   //console.log("eval 300:", s)
   eval(s)
   //console.log("Good :set_acWinStatEventListeners: ", "eval 300")
   } catch (err) {alert("Error 300" + err.message)}
  console.log("set_acWinStatEventListeners :", "Check 0300");

  try{
        var event_frame_ico = new Event("click", {bubbles: false});
        this.win_frame_ico.dispatchEvent(event_frame_ico);
    } catch (err) {console.log("Error set_acWinStatEventListeners :", "Check 0350");}

   console.log("set_acWinStatEventListeners :", "Check 0400");
}

acWin.prototype.remove_win = function()
{
  try{this.win_frame.outerHTML = "";this.win_navL = "";} catch (err) {console.log("Error remove_win :", "remove_win 0450");}
}

function EarningForecast(my_name_, win_name_, win_title_, use_id, tab_obj_)
{
  this.my_name=my_name_;  this.name = "ef_" + win_name_.toString();  this.user_id = user_id; var is_scroll_ = true;
  acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, right="2%", top="30%", is_scroll=is_scroll_, zindex=20, tab_obj_=tab_obj_)
}
EarningForecast.prototype = Object.create(acWin.prototype)


// -- Streamer Object ---
function StreamerWin(my_name_, win_name_, win_title_, use_id, tab_obj_, right, top, onmessage_callback, link_to_activate_obj_function, router, group, is_nav_panel_=false)
{
  this.my_name=my_name_;this.name= "ef_"+win_name_.toString();this.socket=null;
  this.router=router;this.group=group;
  this.link_to_activate_obj_function=link_to_activate_obj_function;this.onmessage_callback=onmessage_callback;
   // course_schedule_id
  this.user_id=user_id; var is_scroll_=true;
  if (vObj["user_group"] == "optionsadmin"){is_nav_panel_=true}
  acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, right=right, top=top, is_scroll=is_scroll_, zindex=20, tab_obj_=tab_obj_, is_nav_panel=is_nav_panel_)
}
StreamerWin.prototype = Object.create(acWin.prototype)

StreamerWin.prototype.getSocket = function(){this.socket = sm.getSocket(this);}

StreamerWin.prototype.socket_onmessage = function(msg){this.process_message(msg);this.onmessage_callback(msg);}

StreamerWin.prototype.socket_onopen = function(e){console.log("socket_onopen", 10000);}

StreamerWin.prototype.send_msg = function (msg){
    var streamer_ = this
    setTimeout(
        function () {
            if (streamer_.socket.readyState === 1) {
                console.log("Connection is made")
                streamer_.socket.send(msg);
                console.log("msg was sent")
            } else {
                console.log("wait for connection...")
                streamer_.send_msg(msg);
            }
        }, 10); // wait 5 milisecond for the connection...
}


// -- Orders --
function OrderStreamerWin(my_name_, win_name_, win_title_, use_id, tab_obj_, data_link, order_link, link_to_activate_obj_function)
{
alert("order 1")
  acWin.call(this, my_name_, win_name_, win_title_, right="2%", top="20%", is_scroll=false, zindex=21, tab_obj_=tab_obj_, is_nav_panel=true)
  this.my_name=my_name_;
  this.user_id=user_id;
  this.ticker=null; this.myInterval_d=null;this.data=null;
  this.data_link=data_link; this.order_link=order_link; this.link_execute_order="";
  this.link_to_activate_obj_function = link_to_activate_obj_function

  this.content_table=document.createElement("table");
  this.win_content.appendChild(this.content_table);

  var tr1 = document.createElement("tr");var tr2 = document.createElement("tr")
  this.content_table.appendChild(tr1); this.content_table.appendChild(tr2)

  delta_content = document.createElement("td");
  // I used K100 as a key to find how to put back delta--
  //
  //K100 tr1.appendChild(delta_content);
  //K100 delta_content.setAttribute("style", "vertical-align: top");

  vertical_content = document.createElement("td");tr1.appendChild(vertical_content);
  vertical_content.setAttribute("style", "vertical-align: top");
  bf_content = document.createElement("td"); tr2.appendChild(bf_content)
  bf_content.setAttribute("colspan", "2");

  this.order = document.createElement("td");tr1.appendChild(this.order);
  //this.order.setAttribute("rowspan", "2");
  this.order.setAttribute("style", "vertical-align: top");

  this.order_nav = document.createElement("div");this.order.appendChild(this.order_nav);
  this.order_nav.onclick = function (event){
    for(j in this.children){try{this.children[j].setAttribute('style', 'color:blue')}catch(er){}}
    var elm = event.target;elm.setAttribute('style', 'color:green')
    var dic = Orders.order_dic["dic"];
    dic["q"] = elm.innerHTML; eval(Orders.order_dic["fun"]+"(dic)");
  }
  var btns = [1,10,20,25,30,40,50,100];
//  1 option (call/put) cost = (0.06 * 100) * 2 (one for enter one for exit)

  for(z in btns){var i=btns[z];eval("this.x"+i+" = document.createElement('button');this.x"+i+".innerHTML="+i+";this.x"+i+".setAttribute('style','color:blue');this.order_nav.appendChild(this.x"+i+");")}

  this.order_content = document.createElement("div");this.order.appendChild(this.order_content);

  this.order_limits = document.createElement("div");this.order.appendChild(this.order_limits);
  this.order_limits.appendChild(document.createElement("br"));
  span_sell_limit = document.createElement("span");this.order_limits.appendChild(span_sell_limit);
  span_sell_limit.innerHTML = "Sell Limit: ";
  this.input_sell_limit = document.createElement("input");this.order_limits.appendChild(this.input_sell_limit);
  this.input_sell_limit.setAttribute("size", 5); this.input_sell_limit.setAttribute("style", "text-align:right");
  this.input_sell_limit.onchange=function(event){
      var elm=event.target; var value=elm.value; Orders.order_dic["limits"]["sell_limit"]=value;
    }
  this.order_limits.appendChild(document.createElement("br"));
  span_buy_limit = document.createElement("span");this.order_limits.appendChild(span_buy_limit);
  span_buy_limit.innerHTML = "Buy Limit: ";
  this.input_buy_limit = document.createElement("input");this.order_limits.appendChild(this.input_buy_limit);
  this.input_buy_limit.setAttribute("size", 5); this.input_buy_limit.setAttribute("style", "text-align:right");
  this.input_buy_limit.onchange=function(event){
      var elm=event.target; var value=elm.value; Orders.order_dic["limits"]["buy_limit"]=value;
    }
  this.set_win_nav_panel();
  this.strikes5=null;
  this.order_dic=null;
  this.strategies={"bf": new BFStrategy(my_name='bf', parent=this, container=bf_content),
                   "vertical": new VerticalStrategy(my_name='vertical', parent=this, container=vertical_content),
                   "delta": new DeltaStrategy(my_name='delta', parent=this, container=delta_content)
                   }

alert("order 1-10")
}
OrderStreamerWin.prototype = Object.create(acWin.prototype)

OrderStreamerWin.prototype.set_win_nav_panel = function()
{
  this.ticker_span=document.createElement("span");
  this.price_span=document.createElement("span");
  //this.price_span.setAttribute("style", "color:green");
  this.spinner_span=document.createElement("span");
  this.date_span=document.createElement("span");
  this.timer_span=document.createElement("span");
  this.timer_span.setAttribute("id", "orders_timer");

  this.win_frame_title_span.appendChild(this.ticker_span);
  this.win_frame_title_span.appendChild(this.price_span);
  this.win_frame_title_span.appendChild(this.date_span);
  this.win_frame_title_span.appendChild(this.timer_span);
  this.win_frame_title_span.appendChild(this.spinner_span);

  if(this.is_nav_panel==true)
  {
     this.stop_start_update = document.createElement("button");
     this.stop_start_update.setAttribute("id", this.my_name+"_stop_start_update");
     this.stop_start_update.setAttribute("style", "width:10%;");
     this.stop_start_update.innerHTML = "Stop"
     this.stop_start_update.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.stop_start_update);
     this.stop_start_update.onclick = function (event) {
      var btn = event.target;var owner_my_name_ = btn.getAttribute("owner_my_name")
      var my_order = eval(owner_my_name_)
      if(btn.innerHTML=="Stop"){btn.innerHTML="Start";if(my_order.myInterval_d !== null){
        clearInterval(my_order.myInterval_d);my_order.myInterval_d=null;
        // my_order.set_msg("");
        };
        //my_order.set_title(my_order.ticker+" <span id='orders_timer'></span>")
      }
      else {btn.innerHTML="Stop";if(my_order.ticker!==null){try{get_orders_data_();my_order.myInterval_d=setInterval(get_orders_data_, 3000)} catch(er){alert(er)}};
      //my_order.set_title(my_order.ticker +"<span id='orders_working'></span><span id='orders_timer'></span>");
      }
    }
     this.confirm_order = document.createElement("button");
     this.confirm_order.setAttribute("id", this.my_name+"_confirm_order");
     this.confirm_order.setAttribute("style", "width:20%;");
     this.confirm_order.innerHTML = "Confirm Order"
     this.confirm_order.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.confirm_order);
     this.confirm_order.onclick = function (event) {
      var btn = event.target;
       var owner_my_name_ = btn.getAttribute("owner_my_name")
       var my_order = eval(owner_my_name_)
       //alert(my_order.my_name)
       alert(JSON.stringify(my_order.order_dic))

       //alert(JSON.stringify(my_order.order_dic["order"]))
       //alert(my_order.order_dic["limits"]["sell_limit"])
       if(my_order.order_dic["limits"]["sell_limit"]=="0"){alert("You must enter sell price.");return;}
       if(my_order.order_dic["limits"]["buy_limit"]=="0"){alert("You must enter buy price and buy price.");return;}
         $.post(my_order.link_to_activate_obj_function,
         {obj: "TDAmeriTrade", fun: "place_order", dic: JSON.stringify(my_order.order_dic)},
          function(data){
           if (data['status'] == 'ok'){
             alert(JSON.stringify(data['data']))
           } else{}
         });
    }
     this.execute_order = document.createElement("button");
     this.execute_order.setAttribute("id", this.my_name+"_execute_order");
     this.execute_order.setAttribute("style", "width:20%;");
     this.execute_order.innerHTML = "Execute Order"
     this.execute_order.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.execute_order);
     this.execute_order.onclick = function (event) {
       var btn = event.target;
       var owner_my_name_ = btn.getAttribute("owner_my_name")
       var my_order = eval(owner_my_name_)
       alert(btn.innerHTML)
       return;

       $.post(my_order.link_execute_order,
       {obj: "TDAmeriTrade", fun: "activate_several_stream", dic: my_order.funcs_to_activate},
        function(data){
         for (k in data){if (k == 'status'){
         alert(data['status'])} else{}
         }
       });
    }
  }
}

OrderStreamerWin.prototype.set_ticker = function(value=""){this.ticker_span.innerHTML = " "+value;}
OrderStreamerWin.prototype.set_spinner = function(value=""){this.spinner_span.innerHTML = value;}
OrderStreamerWin.prototype.set_price = function(value=""){this.price_span.innerHTML = " <b>" + value + "</b>";}
OrderStreamerWin.prototype.set_date = function(value=""){this.date_span.innerHTML = " " + value + " ";}
//OrderStreamerWin.prototype.set_timer = function(value=""){this.timer_span.innerHTML = value;}

OrderStreamerWin.prototype.set_data = function(data)
{
  this.data=data;
  this.set_date(data['date'])
  this.underlyingPrice = Math.round(100*data['underlyingPrice'])/100
  this.set_price(this.underlyingPrice); this.set_spinner("");
  this.strategies["vertical"].set_screen(data["dic13"])
  this.set_msg("");
  this.strategies["delta"].set_screen(data["dic19"])
  this.strategies["bf"].set_screen(data["dic19"])
  this.manage_orders()
}

OrderStreamerWin.prototype.set_orders_screen = function(ticker="")
{
 try{if(this.myInterval_d !== null){this.order_content.innerHTML = "";
        clearInterval(this.myInterval_d); this.myInterval_d = null; //this.set_msg("");
    }} catch(er){}
 this.ticker = ticker; Orders.set_ticker(ticker); Orders.set_price("");
 try{get_orders_data_();this.myInterval_d = setInterval(get_orders_data_, 3000);} catch (er) {alert(er)}
}

function get_orders_data_()
{
  //Orders.set_title(Orders.ticker +" ...  <i class='fa fa-spinner fa-spin'></i><span id='orders_timer'></span>");
  //Orders.set_spinner("...<i class='fa fa-spinner fa-spin'></i>...");
  //var n_ = Orders.get_msg(); if(n_==""){n_=0}; n_=1*n_; n_+=1; Orders.set_msg(n_);
  $.post(Orders.data_link,
    {fun: "get_option_statistics_for_ticker", attribute: 'ticker', attribute_value: Orders.ticker},
    function(data)
    {
       var l_1=0.1;var h_3=1*Orders.strategies["vertical"].delta_percentage/100;var h_9=0.9; var l__999=-1000; var h_999=1000;
       var data = data['option_data_ticker'];
       var arrived_ticker = data['symbol'];
       // var underlyingPrice=Math.round(100*data['underlyingPrice'])/100;

       //Orders.set_price(underlyingPrice); Orders.set_spinner("");
       //Orders.set_title(Orders.ticker + ": " + underlyingPrice + " <span id='orders_timer'></span>");
     if(Orders.ticker==arrived_ticker)
     {  // Prepare the data ----
        types = ["call", "put"]
        dic = {"ticker": Orders.ticker, "dic13": {"put": {}, "call": {}},
                                        "dic19": {"put": {}, "call": {}},
                                        "dic99": {"put": {}, "call": {}}}
        for (i in types)
        {
          var option_type = types[i]
            option_type_ = option_type + 'ExpDateMap'
            for (d in data[option_type_])
            {
                d_ = d.split(":")[0]
                dic['date'] = d_
                dic['underlyingPrice'] = data['underlyingPrice']
                tt = data[option_type_][d]
                for (t in tt)
                {
                    if (tt[t][0]['delta'] != "NaN")
                    {
                      if (l__999 < Math.abs(tt[t][0]['delta']) & Math.abs(tt[t][0]['delta']) < h_999)
                      {
                        if (!(t in dic["dic99"][option_type])){dic["dic99"][option_type][t] = {}}
                        //dic["dic99"][t][option_type] = {};
                        p_ = Math.round(100 * (tt[t][0]['bid'] + tt[t][0]['ask']) / 2) / 100;
                        delta_ = Math.round(100*tt[t][0]['delta']) / 100;
                        theta_ = Math.round(100 * tt[t][0]['theta']) / 100;
                        symbol_ = tt[t][0]['symbol'];
                        dic["dic99"][option_type][t]['price'] = p_;
                        dic["dic99"][option_type][t]['ask'] = tt[t][0]['ask'];
                        dic["dic99"][option_type][t]['bid'] = tt[t][0]['bid'];
                        dic["dic99"][option_type][t]['delta'] = delta_;
                        dic["dic99"][option_type][t]['theta'] = theta_;
                        dic["dic99"][option_type][t]['symbol'] = symbol_;
                        if (l_1 < Math.abs(tt[t][0]['delta']) & Math.abs(tt[t][0]['delta']) < h_9)
                        {
                          if (!(t in dic["dic19"])){dic["dic19"][t] = {}}
                          dic["dic19"][option_type][t] = {}
                          dic["dic19"][option_type][t]['price'] = p_
                          dic["dic19"][option_type][t]['ask'] = tt[t][0]['ask'];
                          dic["dic19"][option_type][t]['bid'] = tt[t][0]['bid'];
                          dic["dic19"][option_type][t]['delta'] = delta_
                          dic["dic19"][option_type][t]['theta'] = theta_
                          dic["dic19"][option_type][t]['symbol'] = symbol_
                          if (l_1 < Math.abs(tt[t][0]['delta']) & Math.abs(tt[t][0]['delta']) < h_3)
                          {
                            if (!(t in dic["dic13"])){dic["dic13"][t] = {}}
                            dic["dic13"][option_type][t] = {}
                            dic["dic13"][option_type][t]['price'] = p_
                            dic["dic13"][option_type][t]['ask'] = tt[t][0]['ask'];
                            dic["dic13"][option_type][t]['bid'] = tt[t][0]['bid'];
                            dic["dic13"][option_type][t]['delta'] = delta_
                            dic["dic13"][option_type][t]['theta'] = theta_
                            dic["dic13"][option_type][t]['symbol'] = symbol_
                          }
                        }
                      }
                    }
                }
            }
        }
        Orders.set_data(data=dic);
      }
     }
    );
}

OrderStreamerWin.prototype.manage_orders = function()
{
  if(this.order_dic==null){return;}
  order = this.order_dic["order"];
  //alert(JSON.stringify(order))
  var cost = 0;var cost_ab = 0;
  var s="<thead><th></th><th>type</th><th>strike</th><th>price</th><th>Qt</th><th>total</th><th>ask</th><th>bid</th><th>ask/bid</th></thead>"

  //alert(JSON.stringify(order))

  for(type in order)
  {
    for(s_ in order[type])
    {
     try{
        var k=order[type][s_]["k"];
        var ll=this.data["dic99"][type][k];   //strategies["vertical"].vertical_data[type];
  //alert(JSON.stringify(ll))

        var q=order[type][s_]["q"]; var p=Math.round(100*ll["price"])/100;
        var ask=Math.round(100*ll["ask"])/100;; var bid=Math.round(100*ll["bid"])/100;
        //alert(q); alert(ask); alert(bid);

        if(q>0){var t_ab=Math.round(100*q*ask)/100; cost_ab+=t_ab} else
        {var t_ab=Math.round(100*q*bid)/100; cost_ab+=t_ab}
        var t=Math.round(100*q*p)/100; cost+=t;

        s+="<tr><td onclick=Orders.remove_strategy_by_type('"+type+"') style='cursor:pointer'>-</td><td>"+type+"</td><td>"+k;
        s+="</td><td style='text-align:right'>"+two_decimal_points(p)+"</td><td style='text-align:right'>"+q;
        s+="</td><td style='text-align:right'>"+two_decimal_points(t)+"</td>";
        s+="</td><td style='text-align:right'>"+two_decimal_points(ask)+"</td>";
        s+="</td><td style='text-align:right'>"+two_decimal_points(bid)+"</td>";
        s+="</td><td style='text-align:right'>"+two_decimal_points(t_ab)+"</td>";
        s+="</tr>";

      } catch (er) {}
    }
  }
 cost=Math.round(100*cost)/100;
 s+="<tr><td></td><td></td><td></td><td></td><td></td><td style='text-align:right'>"+two_decimal_points(cost)+"</td>"
 s+="<td></td><td></td><td style='text-align:right'>"+Math.round(100*cost_ab)/100+"</td>"
 s+="</tr>"
 s = "<table>" + s + "</table>";
 this.order_content.innerHTML = s;
 //alert(JSON.stringify(dic))
}

OrderStreamerWin.prototype.remove_strategy_by_type = function(type)
{
 this.order_dic["order"][type] = {};this.manage_orders();
}

OrderStreamerWin.prototype.zero_sell_buy_limit = function()
{
 this.order_dic["limits"]["sell_limit"]="0";this.order_dic["limits"]["buy_limit"]="0";
 this.input_sell_limit.value =""; this.input_buy_limit.value="";
}

// Strategies
function acStrategy(my_name, parent, container)
{
  this.my_name=my_name;
  this.parent=parent;
  this.container = container;
  this.data=null;
  this.order=null;
}

// BFStrategy --
function BFStrategy(my_name, parent, container)
{
  acStrategy.call(this, my_name, parent, container)
}
BFStrategy.prototype = Object.create(acStrategy.prototype)

BFStrategy.prototype.set_screen = function(data)
{
 this.data = data;
 var strikes5 = this.parent.strikes5;

 var llc =data['call']; var llp = data['put'];
 llc_=[Math.round(100*(llc[strikes5[0]]["price"]-2*llc[strikes5[1]]["price"]+llc[strikes5[2]]["price"]))/100,
       Math.round(100*(llc[strikes5[1]]["price"]-2*llc[strikes5[2]]["price"]+llc[strikes5[3]]["price"]))/100,
       Math.round(100*(llc[strikes5[2]]["price"]-2*llc[strikes5[3]]["price"]+llc[strikes5[4]]["price"]))/100]
 llp_=[Math.round(100*(llp[strikes5[0]]["price"]-2*llp[strikes5[1]]["price"]+llp[strikes5[2]]["price"]))/100,
       Math.round(100*(llp[strikes5[1]]["price"]-2*llp[strikes5[2]]["price"]+llp[strikes5[3]]["price"]))/100,
       Math.round(100*(llp[strikes5[2]]["price"]-2*llp[strikes5[3]]["price"]+llp[strikes5[4]]["price"]))/100]
  s = "<tr><th colspan='6' style='text-align: center'>Butterfly Call</th><th colspan='6' style='text-align: center'>Butterfly Put</th></tr>"
  s += "<tr><th>BF(-1)</th><th>price</th><th>BF(0)</th><th>price</th><th>BF(+1)</th><th>price</th>"
  //s += "<th>SharePrice</th>";
  s += "<th>BF(-1)</th><th>price</th><th>BF(0)</th><th>price</th><th>BF(+1)</th><th>price</th></tr>";
  s += get_row_bf_(num=1, nump=1, strikes5, llc, llp);
  s += get_row_bf_(num=2, nump=-2, strikes5, llc, llp);
  s += get_row_bf_(num=3, nump=1, strikes5, llc, llp);
  s += "<tr><td></td><td style='text-align:right;' type='call' n='0' style='cursor:pointer'>"+two_decimal_points(llc_[0])+"</td><td></td><td style='text-align:right;' type='call' n='1' style='cursor:pointer'>"+two_decimal_points(llc_[1])+"</td><td></td><td style='text-align:right;' type='call' n='2' style='cursor:pointer'>"+two_decimal_points(llc_[2])+"</td>"
  //s += "<td style='text-align: right;'>"+two_decimal_points(share_price)+"</td>";
  s += "<td></td><td style='text-align:right;' type='put' n='0'>"+two_decimal_points(llp_[0])+"</td><td></td><td style='text-align:right;' type='put' n='1' style='cursor:pointer'>"+two_decimal_points(llp_[1])+"</td><td></td><td style='text-align:right;' type='put' n='2' style='cursor:pointer'>"+two_decimal_points(llp_[2])+"</td>"
  s = "<table class='data' onclick=Orders.strategies['bf'].prepare_bf_order_(event) style='cursor:pointer'>" + s + "</table>";
 this.container.innerHTML=s;
}

BFStrategy.prototype.prepare_bf_order_ = function(event)
{
 var elm = event.target;var type=elm.getAttribute("type");var n=elm.getAttribute("n");
 this.prepare_bf_order(dic={"type":type,"n": n,"q":1});
}

BFStrategy.prototype.prepare_bf_order = function(dic=null)
{
 //alert(JSON.stringify(dic));
 var order = {}; order["put"]={};order["call"]={};
 var type=dic["type"];var n__=dic["n"];var q=dic["q"];var n=1*n__;
 var strikes=this.parent.strikes5;var ll=this.data[type];
 var q_=1;var s_=ll[strikes[n]]["symbol"]
 order[type][s_] = {"k":strikes[n], "q": q_}
 n=1*n+1;q_=-2;s_=ll[strikes[n]]["symbol"];
 order[type][s_] = {"k":strikes[n], "q": q_}
 n=1*n+1;q_=1;s_=ll[strikes[n]]["symbol"]
 order[type][s_] = {"k":strikes[n], "q": q_}

 //alert(JSON.stringify(order));

 var order_dic = {"obj_name": this.my_name, "fun": "Orders.strategies['bf'].prepare_bf_order", "limits":{"sell_limit": 0, "buy_limit": 0}, "order": order, "dic": dic}
 //alert(JSON.stringify(order_dic));

 this.parent.order_dic=order_dic; this.parent.manage_orders()
 this.parent.zero_sell_buy_limit();
}

function get_row_bf_(num=1, nump=1, strikes, llc, llp)
{
 var nl=num-1;var nr=num+1; var s = "";
 s += "<tr><td><span>"+strikes[nl]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llc[strikes[nl]]["price"])+"</span></td>"
 s += "<td><span>"+strikes[num]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llc[strikes[num]]["price"])+"</span></td>"
 s += "<td><span>"+strikes[nr]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llc[strikes[nr]]["price"])+"</span></td>"
 //s += "<td style='text-align: right;'>"+share_price+"</td>"
 s += "<td><span>"+strikes[nl]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llp[strikes[nl]]["price"])+"</span></td>"
 s += "<td><span>"+strikes[num]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llp[strikes[num]]["price"])+"</span></td>"
 s += "<td><span>"+strikes[nr]+"</span></td><td style='text-align: right;'><span>"+nump+"*"+two_decimal_points(llp[strikes[nr]]["price"])+"</span></td></tr>"
 //alert(s)
  return s
}


// VerticalStrategy -
function VerticalStrategy(my_name, parent, container)
{
  acStrategy.call(this, my_name, parent, container);
  this.vertical_data = null;
  this.delta_percentage = "30"
}
VerticalStrategy.prototype = Object.create(acStrategy.prototype)

VerticalStrategy.prototype.set_screen = function(data)
{
  this.data = data;
  //alert(JSON.stringify(data))
  var d_=-1;var sk=-1; var sk1=-1; var vertical_data={};
  try{
     var types = ["call", "put"]
     for (i in types)
     {vertical_data[types[i]]={};
      for (k in data[types[i]])
      {if (sk == -1){sk = parseFloat(k)} else {sk1 = parseFloat(k); d_ = Math.abs(sk1 - sk); sk = sk1};
       vertical_data[types[i]][k] = {'delta': data[types[i]][k]['delta'], 'price': data[types[i]][k]['price'], 'symbol': data[types[i]][k]['symbol']}
      }
     }
 } catch(er){alert(er)}
  this.vertical_data = vertical_data;

  // alert(JSON.stringify(vertical_data))
  var sc = "<table><tr><th colspan='3' style='text-align: center'>Vertical Call (<span onclick=Orders.strategies['vertical'].set_delta_range(event)>"+this.delta_percentage+"</span>)</th></tr>"
  sc += "<tr><th>ADelta</th><th>Striks</th><th>Price</th></tr>"
  var n_=0;
  for(k in vertical_data["call"])
  {
   if(n_==0)
   {
    var d_ = vertical_data["call"][k]["delta"];
    var k_ = k;
    var p_ = vertical_data["call"][k]["price"]
    n_=1;
   } else {
    var d = vertical_data["call"][k]["delta"];
    var p = vertical_data["call"][k]["price"];
    var ad_1="<td>"+((Math.round(100*(d_+d))/100)/2).toString()+"</td>"
    var ss_1="<td onclick=Orders.strategies['vertical'].prepare_vertical_order_(event) type='call' style='cursor:pointer'>"+k_+"/"+k+"</td>";
    var p_1 ="<td>"+(Math.round(100*(p_-p))/100).toString() +"</td>";
    sc+= "<tr>"+ad_1+ss_1+p_1+"</tr>"
    d_ = d; k_ = k; p_ = p
   }
 }
  var sp = "<table><tr><th colspan='3' style='text-align: center'>Vertical Put (<span onclick=Orders.strategies['vertical'].set_delta_range(event)>"+this.delta_percentage+"</span>)</th></tr>"
  sp += "<tr><th>ADelta</th><th>Striks</th><th>Price</th></tr>"
  var n_=0;
  for(k in vertical_data["put"])
  {
   if(n_==0)
   {
    var d_ = vertical_data["put"][k]["delta"];
    var k_ = k;
    var p_ = vertical_data["put"][k]["price"]
    n_=1;
   } else {
    var d = vertical_data["put"][k]["delta"];
    var p = vertical_data["put"][k]["price"];
    var ad_1="<td>"+((Math.round(100*(d_+d))/100)/2).toString()+"</td>"
    var ss_1="<td onclick=Orders.strategies['vertical'].prepare_vertical_order_(event) type='put' style='cursor:pointer'>"+k_+"/"+k+"</td>";
    var p_1 ="<td>"+(Math.round(100*(p-p_))/100).toString() +"</td>";
    sp+= "<tr>"+ad_1+ss_1+p_1+"</tr>"
    d_ = d; k_ = k; p_ = p
   }
 }
  var s = "<div style='height:500px;width:500px;overflow:scroll;'><table class='data'><tr><td style='vertical-align: top'>"+sc+"</table></td><td style='vertical-align: top'>"+sp+"</table></td></table></div>"

  this.container.innerHTML=s;
}

VerticalStrategy.prototype.prepare_vertical_order_ = function(event)
{
 var elm = event.target; var type = elm.getAttribute("type");
 var ks = elm.innerHTML; var ks_=ks.split("/")
 this.prepare_vertical_order(dic={"k0": ks_[0], "k1": ks_[1], "type": type, "q":1});
}

VerticalStrategy.prototype.prepare_vertical_order = function(dic=null)
{
 //alert(JSON.stringify(dic))
 var type = dic["type"];
 // var q=dic["q"];
 var q=1;
 if(type=="call"){var k0=dic["k0"]; var k1=dic["k1"];} else {var k0=dic["k1"]; var k1=dic["k0"];}
 var ll=this.vertical_data[type];

 if(this.parent.order_dic!=null && this.my_name == this.parent.order_dic["obj_name"])
 {order=this.parent.order_dic["order"];order[type]={}} else {var order={};order["put"]={};order["call"]={};}

 var s_=ll[k0]["symbol"]; var q_=1*q;var p_=Math.round(100*ll[k0]["price"])/100; var t_=Math.round(100*q_*p_)/100;
 order[type][s_] = {"k":k0, "q": q_} //, "p":p_, "t":t_}
 var q_=-1*q;var p_=two_decimal_points(ll[k1]["price"]); var s_=ll[k1]["symbol"]; var t_=Math.round(100*q_*p_)/100;
 order[type][s_] = {"k":k1, "q": q_} //, "p":p_, "t":t_}

 // -- this makes sure that both verticals will have the same quantity.
 // Remove this line and each vertical will have its own quantity.
 for (type_ in order){for (s__ in order[type_]){var h=order[type_][s__]["q"]/Math.abs(order[type_][s__]["q"]);order[type_][s__]["q"] = h*q;}}
 // -------------------------

 // alert(JSON.stringify(order));
 var order_dic = {"obj_name": this.my_name, "fun": "Orders.strategies['vertical'].prepare_vertical_order", "limits":{"sell_limit": 0, "buy_limit": 0}, "order": order, "dic": dic}
 this.parent.order_dic=order_dic; this.parent.manage_orders();
 this.parent.zero_sell_buy_limit();
}


VerticalStrategy.prototype.set_delta_range = function(event)
{
  var elm = event.target;var p=elm.innerHTML;
  if(p=="30"){p="40"} else if (p=="40"){p="100"} else{p="30"};elm.innerHTML=p;this.delta_percentage=p;



}

// DeltaStrategy --
function DeltaStrategy(my_name, parent, container)
{
  acStrategy.call(this, my_name, parent, container)
}
DeltaStrategy.prototype = Object.create(acStrategy.prototype)

DeltaStrategy.prototype.set_screen = function(data)
{
 this.data = data;
 types = ["call", "put"]
 var data_tickers=[];
 for (i in types){var option_type=types[i];for (z in data[option_type]) {data_tickers.push(z)}}
 data_tickers = [...new Set(data_tickers)]
 data_tickers = data_tickers.sort(function(a, b) {return a - b;});
 var d_=10000; var kn_=0;
 var underlyingPrice = this.parent.underlyingPrice
 for(i in data_tickers){var d=Math.abs(underlyingPrice-data_tickers[i]); if(d<d_){d_=d;kn_=data_tickers[i]}}
 n_=data_tickers.indexOf(kn_)
 var n_2 = n_-2;var n_1=n_-1;var n2=n_+2; var n1=n_+1;
 this.parent.strikes5 = [data_tickers[n_2], data_tickers[n_1], data_tickers[n_], data_tickers[n1], data_tickers[n2]]

 //K100 to remove delta --
 return;

 var s = "<tr><th colspan='3' style='text-align: center'>Call</th><th style='background-color:green'>"+underlyingPrice+"</th>"
 s += "<th colspan='3' style='text-align: center'>Put</th></tr>"
 s += "<tr><th>Delta</th><th>Theta</th><th>Price</th><th>Strike</th><th>Price</th><th>Theta</th><th>Delta</th></tr>"
 for (ii in data_tickers)
 {
  var z = data_tickers[ii];
  var st = "<td style='text-align: right;'>" + z + "</td>"
  try{
     d = two_decimal_points(dic["dic19"]['call'][z]['delta'])
     t = two_decimal_points(dic["dic19"]['call'][z]['theta'])
     p = two_decimal_points(dic["dic19"]['call'][z]['price'])
     if(1*underlyingPrice>1*z){ var zz= " class = 'inTheMoney'"}else {var zz=""}
     st = "<td style='text-align: right;'"+zz+">"+d+"</td><td style='text-align: right;'"+zz+">"+t+"</td><td style='text-align: right;'"+zz+">"+p+"</td>" + st
     } catch(er){
     st = "<td></td><td></td><td></td>" + st}
  try{
     d = two_decimal_points(dic["dic19"]['put'][z]['delta'])
     t = two_decimal_points(dic["dic19"]['put'][z]['theta'])
     p = two_decimal_points(dic["dic19"]['put'][z]['price'])
     if(1*underlyingPrice<=1*z){ var zz= " class = 'inTheMoney'"} else {var zz=""}
     st += "<td style='text-align: right;'"+zz+">"+p+"</td><td style='text-align: right;'"+zz+">"+t+"</td><td style='text-align: right;'"+zz+">"+d+"</td>"
     } catch(er){
     st += "<td></td><td></td><td></td>"};
     s += "<tr>" + st + "</tr>"
}
 this.container.innerHTML="<div style='height:500px;width:440px;overflow:scroll;'><table class='data'>"+s+"</table></div>";
 //var n_ = Orders.get_msg(); if(n_==""){n_=0}; n_=1*n_; n_-=1; Orders.set_msg(n_);
}


// -- OptionStreamerWin --
function OptionStreamerWin(my_name_, win_name_, win_title_, use_id, tab_obj_, onmessage_callback, link_to_activate_obj_function, router, group, tickers, funcs_to_activate)
{
  this.funcs_to_activate=funcs_to_activate;
  this.tickers=tickers;
  StreamerWin.call(this, my_name_, win_name_, win_title_, use_id, tab_obj_, right="2%", top="20%", onmessage_callback, link_to_activate_obj_function, router, group, funcs_to_activate);
  this.container=null;
  this.initialDateStr = '6 Jan 2022 21:28 GMT';
  this.initialDate = luxon.DateTime.fromRFC2822(this.initialDateStr);
  this.lastDate = this.initialDate;
  this.set_options_input_tickers();
}
OptionStreamerWin.prototype = Object.create(StreamerWin.prototype)

OptionStreamerWin.prototype.set_options_input_tickers = function()
{
  if(this.is_nav_panel==true)
  {
     this.options_input_tickers = document.createElement("input");
     this.options_input_tickers.setAttribute("id", "options_input_tickers");
     this.options_input_tickers.setAttribute("value", this.tickers);
     this.options_input_tickers.setAttribute("style", "width:70%;");
     this.options_button_get_tickers = document.createElement("button");
     this.options_button_get_tickers.setAttribute("id", "options_button_get_tickers");
     this.options_button_get_tickers.setAttribute("style", "width:10%;");
     this.options_button_get_tickers.innerHTML = "Get Data"
     this.options_button_get_tickers.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.options_input_tickers);
     this.win_nav_panel.appendChild(this.options_button_get_tickers);
     this.options_button_get_tickers.onclick = function (event) {
       var owner_my_name_ = event.target.getAttribute("owner_my_name")
       var my_streamer = eval(owner_my_name_)
       $.post(my_streamer.link_to_activate_obj_function,
       {obj: "TDAmeriTrade", fun: "activate_several_stream", dic: my_streamer.funcs_to_activate},
        function(data){
         for (k in data){if (k == 'status'){
          // alert(data['status'])
         } else{}}
       });
    }
     // ----
     this.options_button_data = document.createElement("button");
     this.options_button_data.setAttribute("id", "options_button_data");
     this.options_button_data.setAttribute("style", "width:10%;");
     this.options_button_data.innerHTML = "TestData"
     this.options_button_data.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.options_button_data);
     this.options_button_data.onclick = function (event) {
        var owner_my_name_ = event.target.getAttribute("owner_my_name")
        var my_streamer = eval(owner_my_name_);
        my_streamer.set_data_for_tickers();
     }
     //-
     this.options_button_one_data = document.createElement("button");
     this.options_button_one_data.setAttribute("id", "options_button_one_data");
     this.options_button_one_data.setAttribute("style", "width:10%;");
     this.options_button_one_data.innerHTML = "OneData"
     this.options_button_one_data.setAttribute("owner_my_name", this.my_name);
     this.win_nav_panel.appendChild(this.options_button_one_data);
     this.options_button_one_data.onclick = function (event) {
        var owner_my_name_ = event.target.getAttribute("owner_my_name")
        var my_streamer = eval(owner_my_name_);
        var msg_ = {}
        my_streamer.lastDate = my_streamer.lastDate.plus({minutes: 1});
            for (var j in my_streamer.csm.objs)
            {
             var d = randomBar(my_streamer.lastDate);
             var ll = [d['x'],d['o'],d['h'],d['l'],d['c']]
             msg_[j] = ll
            }
        msg__ = {"type": "data_received_chart_equity", "msg": msg_}
        //alert(JSON.stringify(msg__))
        my_streamer.process_message(msg=msg__);
     }
  }

 // --
 this.win_content.innerHTML = "<table><thead><th>Chart</th><th>Strategy</th></thead><tbody id='"+this.my_name+"_options'></tbody></table>"
 this.container = document.getElementById(this.my_name+"_options")
 this.csm = new CandlestickChartManager(option_streamer=this)
}

// -- for testing --
OptionStreamerWin.prototype.set_data_for_tickers = function()
{
 for(var k in this.csm.objs)
 {
   var chart_obj_=this.csm.objs[k]; this.data_function(chart_obj_);
   chart_obj_.chart.config.data.datasets = [{label: chart_obj_.label, data: chart_obj_.data}]
   // alert(JSON.stringify(chart_obj_.chart.config.data.datasets))
   chart_obj_.chart.update();
 }
}

OptionStreamerWin.prototype.data_function = function(chart_obj, count=3) {
  var data = [randomBar(chart_obj.parent.parent.initialDate)];
  while (data.length < count) {
    chart_obj.parent.parent.lastDate = chart_obj.parent.parent.lastDate.plus({minutes: 1});
    if (chart_obj.parent.parent.lastDate.weekday <= 5) {
        data.push(randomBar(chart_obj.parent.parent.lastDate));
    }
}
chart_obj.data = data
}

function CandlestickChartManager(option_streamer)
{
 this.parent=option_streamer; this.objs={};
 var ll_tickers = this.parent.tickers.split(",")
 for(i in ll_tickers)
 {
  var t = ll_tickers[i];
  var tr_ = document.createElement("tr");
  tr_.setAttribute("id", "host_row_"+t);
  var td = document.createElement("td");
  var td_div = document.createElement("div");
  td_div.setAttribute("id", "host_div_"+t);
  td_div.setAttribute("style", "width:1100px;height:150px");
  td.appendChild(td_div);
  tr_.appendChild(td);
  td = document.createElement("td");
  td.setAttribute("id", "analysis_"+t);
  td.setAttribute("ticker", t);
  td.setAttribute("style", "cursor:pointer");
  td.innerHTML=t;tr_.appendChild(td);
  td.onclick = function analysis_of_ticker(){
     var e=event.target;var ticker=e.getAttribute("ticker");Orders.set_orders_screen(ticker)
  }.bind(elm = this, ticker=t, event)
  this.parent.container.appendChild(tr_);
  this.objs[t] = new CandlestickChart(this, t, td_div)
 }
}

function CandlestickChart(csm, t, td_div)
{
 this.parent=csm;this.ticker=t;this.host_div=td_div;
 this.canvas = document.createElement("canvas");
 this.canvas.setAttribute("id", "canvas_"+this.ticker);
 this.host_div.appendChild(this.canvas);
 this.ctx = this.canvas.getContext('2d');
 this.ctx.canvas.width = this.host_div.style.width;
 this.ctx.canvas.height = this.host_div.style.height;
 this.data = []
 this.label = 'Chart for '+this.ticker
 this.chart = new Chart(this.ctx, {type: 'candlestick', data: {datasets: [{label: this.label, data: this.data}]}});
 this.chart.update()
}

OptionStreamerWin.prototype.process_message = function(msg)
{
try{

//alert(JSON.stringify(msg))
//alert(msg.type)

  //if(vObj["user_group"] == "optionsadmin")
  //{
    //alert(JSON.stringify(msg))
  //}
 // if(msg.type=="data_received_nasdaq_order_book")
 // {var msg=JSON.parse(msg.msg);
 //  for(k in msg) {
 //    this.add_info_by_ticker(k + " " + msg[k]["OPEN_PRICE"] + " " + msg[k]["CLOSE_PRICE"] + "<br/>", "Option Stream - Data Arrived")
 //  }; this.add_info_by_ticker("<br/>", "Option Stream - Data Arrived")
 // } else if (msg.type=="data_received_option_order_book")
 // {this.add_info_by_ticker(msg.type + " " + msg.msg + "<br/>", "Option Stream - option_order_book")}

  if (msg.type=="connected")
  {

  }
  else if (msg.type=="data_received_chart_equity")
  {
        var msg=msg.msg
        for(k in msg){
         var date_ = msg[k][0]
         //var date_ = luxon.DateTime.fromRFC2822(msg[k][0]);
         var point_data={x: date_.valueOf(), o: msg[k][1], h: msg[k][2], l: msg[k][3], c: msg[k][4]};
         //alert(JSON.stringify(point_data))
         //alert('JSON.stringify(this.csm.objs[k].data)')
         //alert(JSON.stringify(this.csm.objs[k].data))
         this.csm.objs[k].data.push(point_data);
         //alert(JSON.stringify(this.csm.objs[k].data))
         this.csm.objs[k].chart.config.data.datasets = [{label: this.csm.objs[k].label, data: this.csm.objs[k].data}]
         this.csm.objs[k].chart.update()
        };
  } else if (msg.type=="connected"){
    //this.add_info_by_ticker("", "Option Stream - Connected")
  }
  else
  {
    // this.add_info_by_ticker(msg.type + " " + msg.msg + "<br/>", "Option Stream - Connected")
  }
 } catch (er) {alert(er)}
}

function randomBar(date) {
    var basePrice = 1000
    var open = +randomNumber(basePrice * 0.95, basePrice * 1.05).toFixed(2);
    var close = +randomNumber(open * 0.95, open * 1.05).toFixed(2);
    var high = +randomNumber(Math.max(open, close), Math.max(open, close) * 1.05).toFixed(2);
    var low = +randomNumber(Math.min(open, close) * 0.95, Math.min(open, close)).toFixed(2);
    return {x: date.valueOf(),o: open,h: high,l: low,c: close};
}
function randomNumber(min, max) {return Math.random() * (max - min) + min;}
