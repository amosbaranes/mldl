<!--R and D-->
function click_main_randd(event, game_id)
{
  elm = event.target
  obj_id = elm.getAttribute("obj_id")
  type = elm.getAttribute("type")
  if (type == "product")
  {
        var product_id = elm.getAttribute("product_id")
        if (product_id == -1)
        {
            if (elm.getAttribute("creation_period") == elm.getAttribute("game_period"))
            { alert("You can not create a project this period!"); return}

            elm.setAttribute("id", "NEW_OBJ_NEED_MODIFICATION")
            $.post(link1,
                {
                  app: "globsim",
                  model_name: "randd",
                  obj_id: obj_id,
                  product_id: product_id
                },
                function(data){
                  product_id = data['product_id']
                }).done(function() {
                sm.TabObjects["Product"].need_update_screen = 1;
                sm.TabObjects["Product"].CONTENTObjects = {};
                set_product("pr_" + product_id)

                elm = document.getElementById("NEW_OBJ_NEED_MODIFICATION")
                elm.setAttribute("id", "rd_pr_"+product_id)
                elm.setAttribute("product_id", product_id)
                elm = document.getElementById("rd_pr_"+product_id)
                new_obj = document.getElementById("pr_" + product_id)
 //               try{elm.innerHTML = new_obj.innerHTML} catch (e) {alert(e)}
//                s_r = "document.getElementById('rd_pr_" + product_id + "').innerHTML=document.getElementById('pr_" + product_id + "').innerHTML"
//                sm.TabObjects["Product"].run_code_insure_obj(product_id, s_r)
            });
        } else {set_product("pr_" + product_id)}
    return;
  }
  sm.TabObjects["RandD"].click_main(obj_id);
  if (obj_id == "new")
  {
      sm.TabObjects["RandD"].callback_new = (div_, obj_) =>
      {
        obj_id_ = div_.getAttribute("pkey")
        div_.setAttribute("id", "rd_obj_" + obj_id_)
        rd = document.getElementById("randd_detail_attributes_"+obj_id_)
        period = rd.getAttribute("period");
        name = rd.getAttribute("obj_name");
        rd_projects = document.getElementById("randd_projects_");
        var r = rd_projects.children[0].cloneNode(true);
        rd_projects.appendChild(r);
        r.children[0].innerHTML = period
        r.children[1].setAttribute("obj_id", obj_id_)
        r.children[1].setAttribute("id", "rd_"+obj_id_)
        r.children[1].setAttribute("style", "color:blue")
        r.children[1].innerHTML = name
        r.children[2].setAttribute("id", "rd_pr_-1")
        r.children[2].setAttribute("obj_id", obj_id_)
        r.children[2].setAttribute("product_id", "-1")
        r.children[2].setAttribute("creation_period", period)
        r.children[2].innerHTML = "Assign..."
        rd_projects.appendChild(r);
      }
  }
}

function set_product(obj_name)
{
    sm.TabObjects["Product"].is_show_first = false;
    var event = new Event("click", {bubbles: true});
    sm.TabObjects["Product"].tabTitle.dispatchEvent(event);
    setTimeout(set_first_obj, 1, "product", null, obj_name)
}

function slider(event) {
  event.preventDefault();
  elm = event.target
  num = elm.getAttribute("num")
  var output = document.getElementById("input_"+num);
  output.value = elm.value;
}

function slider_change(event, link) {
  event.preventDefault();
  elm = event.target
  num = elm.getAttribute("num")
  update_attribute_value(elm.value, num, link);
}

function input(event, link) {
  event.preventDefault();
  elm = event.target
  num = elm.getAttribute("num")
  var slider = document.getElementById("slider_"+num);
  slider.value = elm.value;
  update_attribute_value(elm.value, num, link);
}

function update_attribute_value(value, num, link) {
      $.post(link,
        {
          value: value,
          obj_id: num
        },
        function(data){
            if(data["status"]=="ok")
            {
              //alert(data["index"])
              var index = document.getElementById("index_"+num);
              index.innerHTML = "("+data["index"]+")"
              }
        }
      );
}

function change_segment(event, link)
{
  e = event.target;
  o = e.options[e.selectedIndex]
  obj_id = o.getAttribute("num")
  slug = o.value;
    $.post(link,
      {
        app: "globsim",
        model_name: "randd",
        obj_id: obj_id,
        slug: slug
      },
      function(data){
        document.getElementById("randd_detail_attributes_"+obj_id).innerHTML = data
      }
    );
}
