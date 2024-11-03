<!-- game_setup -->
function click_main_game_setup(event, slug)
{
    elm = event.target
    fun = elm.getAttribute("fun")
    if (fun == "schedule_period_dates")
    {
        doc_ = sm.TabObjects["GameSetup"].CONTENT;
        $.post(link_management_data,
          {
            slug: slug,
            fun: fun
          },
          function(data){
            doc_.innerHTML = data
          })
    }
    else if (fun == "students_list")
    {
        slug = elm.getAttribute("slug")
        doc_ = sm.TabObjects["GameSetup"].CONTENT;
        $.post(link_students_list,
          {
            slug: slug
          },
          function(data){
            doc_.innerHTML = data
          })
    }
    else if (fun == "random_group_assignment")
    {
        slug = elm.getAttribute("slug")
        doc_ = sm.TabObjects["GameSetup"].CONTENT;
        $.post(link_course_schedule_random_group_assignment_,
          {
            slug: slug
          },
          function(data){
            doc_.innerHTML = data
          })
    }
    else
    {
        $.post(link_management,
          {
            slug: slug,
            fun: fun
          },
          function(data){
            var ss = data['status']
            alert(ss)
          })
    }
}


function click_main_games_admin(event, slug)
{
  elm = event.target
// alert(elm.outerHTML);
    fun = elm.getAttribute("fun")
    if (fun == null) { return }
    obj_id = ""; obj_slug = "";
    try{obj_id = elm.getAttribute("obj_id")} catch (er) {}
    try{obj_slug = elm.getAttribute("obj_slug")} catch (er) {}
    doc_ = sm.TabObjects["GamesAdmin"].CONTENT;
//    alert(link_management_data)
    $.post(link_management_data,
      {
        slug: slug,
        fun: fun,
        obj_slug: obj_slug,
        obj_id: obj_id
      },
      function(data){
        doc_.innerHTML = data
      })
}


function change_game_types(event, link)
{
    e = event.target;
    o = e.options[e.selectedIndex]
    game_id = o.getAttribute("num")
    slug = o.value;
    $.post(link,
      {
        app: "globsim",
        model_name: "game",
        game_id: game_id,
        slug: slug
      },
      function(data){ alert(data['status'])}
    );
}

function update_num_of_periods(event)
{
    elm = document.getElementById("setup_")
    var event = new Event("click", {bubbles: true});
    elm.dispatchEvent(event);
}