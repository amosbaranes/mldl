<!-- General -->
function update_field_model(event, link, call_back="")
{
   elm = event.target
//   alert(elm.outerHTML)
   app_= elm.getAttribute("app")
   model_= elm.getAttribute("model")
   pkey_= elm.getAttribute("pkey")
   type_= elm.getAttribute("type")
   obj_slug = ""
   try {obj_slug = elm.getAttribute("obj_slug")} catch (er) {}
   if (type_ == "checkbox")
   {
    value_ = elm.checked
   } else{
    value_= elm.value
   }
   min = parseInt(elm.getAttribute("min")); max = parseInt(elm.getAttribute("max"))
   if (max != null && min != null)
   {if ( parseInt(value_) < min || parseInt(value_) > max)
     {alert("value must be between " + min.toString() + " and " + max.toString());
      if(parseInt(value_) < min){elm.value = min} else {elm.value = max};
      var event = new Event("change", {bubbles: true});elm.dispatchEvent(event);
      return;
     }
   }
   field_= elm.getAttribute("field")
  $.post(link,
    {
      app: app_,
      model: model_,
      pkey: pkey_,
      value: value_ ,
      field: field_ ,
      type: type_ ,
      obj_slug: obj_slug ,
    },
    function(data){
//        alert(data['status'])
       try {eval(call_back)} catch(err) { alert(err)}
       try {
            update_= elm.getAttribute("update")
            update = update_.split(";")
            for( u in update)
            {
             u_ = update[u]
             rdelm = document.getElementById(u_)
             rdelm.innerHTML=value_
            }
           }
       catch(err) {
            // alert(err)
           }
    }
  );
}


log_info = document.getElementById("log_info")
log = function (l)
{
    log_info.innerHTML = log_info.innerHTML + "--- " + l + "---"
}
