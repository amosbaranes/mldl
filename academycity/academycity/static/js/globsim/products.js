<!-- Products -->
function click_main_product(event)
{
  obj_id = event.target.getAttribute("obj_id")
  sm.TabObjects["Product"].click_main(obj_id)
}

function click_abandon(event)
{
  e = event.target
//  alert(e.outerHTML)
  game_id = e.getAttribute("game_id")
  obj_id = e.getAttribute("obj_id")
  is_abandon = e.getAttribute("is_abandon")
  if (is_abandon == 1){ return -1;}
  {response = prompt('Are you sure you want to abandon this product?', 'NO');
   if(response == "NO" || response == null) {return -1;}
  }
    $.post(link_abandon_product,
      {
        game_id : game_id,
        obj_id : obj_id,
        is_abandon: is_abandon
      },
      function(data){
      try {
//          alert(data['status'])
          e.setAttribute("is_abandon", "1")
          e.innerHTML = ""
          name_ = sm.TabObjects["Product"].name.toLowerCase()
          document.getElementById(name_ + '_obj_' + obj_id).outerHTML=""
          delete  sm.TabObjects["Product"].CONTENTObjects[obj_id]
          set_product("pr_" + obj_id.toString())
          } catch (er)
          {
//          alert(er)
          }
      });

  e.preventDefault();

}


function change_randd(event, link)
{
  e = event.target;
  o = e.options[e.selectedIndex]
  obj_id = o.getAttribute("num")
  randd_pdd_id = o.getAttribute("randd_pdd_id")
  randd_id = o.value;
    $.post(link,
      {
        app: "globsim",
        model_name: "product",
        obj_id: obj_id,
        randd_id: randd_id,
        randd_pdd_id: randd_pdd_id
      },
      function(data){
        if(data['status'] != "ok")
        {alert("Error updating R and D")}
//        document.getElementById("randd_detail_attributes_"+obj_id).innerHTML = data
      }
    );
}




