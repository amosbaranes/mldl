
var csrftoken = Cookies.get('csrftoken');
  function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
  }
  $.ajaxSetup({
    beforeSend: function(xhr, settings) {
      if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
        xhr.setRequestHeader("X-CSRFToken", csrftoken);
      }
    }
  });

var slides;
var dots;
var slug_ = '' ;


// menu functions
var towns_menu_content = {}
function menu_onclick(event, slug)
{

   //event.preventDefault();
   var elm = event.target;

   var menu = elm.getAttribute("menu");
   var town_content = document.getElementById("town_content")
   if(slug_ != ''){slug = slug_;};
   // alert(menu +" : " + slug)

   if(menu == 'home')
   {
     let home_link = document.getElementById('home_'+slug)
     let home_link_event = new Event("click", {bubbles: true});
     home_link.dispatchEvent(home_link_event);
   } else if (menu == 'close_open_menu')
   {
      var to_do = elm.getAttribute("to_do");
      if(to_do == "close")
      {open_close_menu(is_open=false);elm.setAttribute("to_do", "open");}
      else
      {open_close_menu(is_open=true);elm.setAttribute("to_do", "close");}
   } else if (menu == "open_gov" || menu == "faqs")
   {

    // alert(menu)

   } else
   {
      var dic = {
          menu_: menu,
          town_slug: slug
      }
      var key_ = menu+slug
      try{
         dic['item_slug'] = elm.getAttribute("item_slug");
         var key_ = elm.getAttribute("item_slug");
         } catch (err) {alert(err.message)}

         var data = towns_menu_content[key_];
         if (data != null)
         {
           town_content.innerHTML = data;
           return;
         }

         //alert(dic['menu_'])
         //alert(dic['town_slug'])
         //alert(dic['item_slug'])

      $.post(link_menu_town,
        dic,
        function(data){
          town_content.innerHTML = data;
        }
      );
   }
}

function send_contact_us(event)
{
  event.preventDefault();
  var elm = event.target
  var slug_ = elm.getAttribute('slug')
  if (slug_ == null){return;}
  var contact_us_name_ = document.getElementById("contact_us_name").value
  var contact_us_email_ = document.getElementById("contact_us_email").value
  var contact_us_subject_ = document.getElementById("contact_us_subject").value
  var contact_us_message_ = document.getElementById("contact_us_message").value
  $.post(link_post_contact_us,
    {
      slug : slug_,
      contact_us_name : contact_us_name_,
      contact_us_email : contact_us_email_,
      contact_us_subject : contact_us_subject_,
      contact_us_message : contact_us_message_
    },
    function(data){
     alert(data['status'])
    }
  ).fail(function(error) { alert(error.responseJSON) });
}

// set town info
var towns_content = {}
function show_town(event)
{
     var elm = event.target
     var town_content = document.getElementById("town_content")
     if (elm.getAttribute("id") == "mySearch"){event.preventDefault();return;}
     slug_ = elm.getAttribute('slug');
     var data = towns_content[slug_];

     if (data != null)
     {
       town_content.innerHTML = data;
       set_town_info_();

       var elm_close_open_btn = document.getElementById("close_open_btn")
       let elm_close_open_btn_event = new Event("click", {bubbles: true});
       elm_close_open_btn.dispatchEvent(elm_close_open_btn_event);

       return;
     }
      $.post(link_basic_town,
        {
          slug: slug_
        },
        function(data){

          town_content.innerHTML = data;
          towns_content[slug_] = data;
          set_town_info_();

         var elm_close_open_btn = document.getElementById("close_open_btn")
         let elm_close_open_btn_event = new Event("click", {bubbles: true});
         elm_close_open_btn.dispatchEvent(elm_close_open_btn_event);

          //slides = document.getElementsByClassName("mySlides");
          //dots = document.getElementsByClassName("dot");
         // showSlides();
        }
      );
}


function set_town_info_()
{
 var elm = document.getElementById("town_info_")

 var s_town_logo = elm.getAttribute("town_logo")
 var s_town_name = elm.getAttribute("town_name")
 var s_town_mayo = elm.getAttribute("town_mayor")
 var s_town_clerk = elm.getAttribute("town_clerk")

 var s_town_name = elm.getAttribute("town_name")
 var s_address1 = elm.getAttribute("address1")
 var s_address2 = elm.getAttribute("address2")
 var s_phone = elm.getAttribute("phone")
 var s_email = elm.getAttribute("email")

 var s_media_ticker = elm.getAttribute("media_ticker")
 var s_bottom_info_color = elm.getAttribute("bottom_info_color")

 var s = '<div class="row">'
 s += '<div style="float: left;width: 15%; padding: 10px;">'
 s += '   <img id="town_logo" src="'+s_town_logo+'" width="70px" class="rounded-circle">'
 s += '</div>'
 s += '<div style="float: left;width: 85%; padding: 10px;font-size:20px;">'
 s += s_town_name +' Municipal Council<br/>'
 s += 'Mayor: ' + s_town_mayo +'<br/>'
 s += 'Town Clerk: ' + s_town_clerk +'<br/>'
 s += '</div>'
 s += '</div>'

 var elm_to = document.getElementById("town_info")
 elm_to.innerHTML = s

 s = s_town_name + ' Municipal Council<br/>'
 s += s_address1 + ' <br/>'
 s += s_address2 + ' <br/>'
 s += 'Tel: ' + s_phone + ' <br/>'
 s += 'Email: ' + s_email + ' <br/>'
 var elm_town_contact_address_ = document.getElementById("town_contact_address_")
 town_contact_address_.innerHTML = s

 s = '&nbsp;<a href="https://facebook.com/' + s_media_ticker + '" target="_blank"><em class="fa fa-facebook" aria-hidden="true"></em></a>'
 s += '&nbsp;<a href="//twitter.com/' + s_media_ticker + '" target="_blank"><em class="fa fa-twitter" aria-hidden="true"></em></a>'
 s += '&nbsp;<a href="//youtube.com/' + s_media_ticker + '" target="_blank"><em class="fa fa-youtube" aria-hidden="true"></em></a>'
 s += '&nbsp;<a href="//instagram.com/' + s_media_ticker + '" target="_blank"><em class="fa fa-instagram" aria-hidden="true"></em></a>'
 s += '&nbsp;<a href="//flickr.com/' + s_media_ticker + '" target="_blank"><em class="fa fa-flickr" aria-hidden="true"></em></a>'

 var elm_town_contact_media_ = document.getElementById("town_contact_media_")
 elm_town_contact_media_ = s

 document.getElementById("media_column").style.color = s_bottom_info_color

 weatherBalloon( city_name = s_town_name );

 var elm_projects_menus_ = document.getElementById("projects_menus_")
 var elm_projects_menus = document.getElementById("projects_menus")
 elm_projects_menus.innerHTML = elm_projects_menus_.innerHTML

 var elm_directors_menus_ = document.getElementById("directors_menus_")
 var elm_directors_menus = document.getElementById("directors_menus")
 elm_directors_menus.innerHTML = elm_directors_menus_.innerHTML

 var elm_services_menus_ = document.getElementById("services_menus_")
 var elm_services_menus = document.getElementById("services_menus")
 elm_services_menus.innerHTML = elm_services_menus_.innerHTML

 var elm_newannouncement_menus_ = document.getElementById("newannouncement_menus_")
 var elm_newannouncement_menus = document.getElementById("newannouncement_menus")
 elm_newannouncement_menus.innerHTML = elm_newannouncement_menus_.innerHTML

 var elm_careers_menus_ = document.getElementById("careers_menus_")
 var elm_careers_menus = document.getElementById("careers_menus")
 elm_careers_menus.innerHTML = elm_careers_menus_.innerHTML

 var elm_tourism_menus_ = document.getElementById("tourism_menus_")
 var elm_tourism_menus = document.getElementById("tourism_menus")
 elm_tourism_menus.innerHTML = elm_tourism_menus_.innerHTML

}

function search(event) {
  var input, filter, ul, li, a, i;
  input = document.getElementById("mySearch");
  filter = input.value.toUpperCase();
  ul = document.getElementById("myMenu");
  li = ul.getElementsByTagName("a");
  for (i = 0; i < li.length; i++) {
    a = li[i];
    if (a.innerHTML.toUpperCase().indexOf(filter) > -1) {
      li[i].style.display = "";
    } else {
      li[i].style.display = "none";
    }
  }
}

// slide show -- not used to be deleted
var slideIndex = 0;
function showSlides() {
  var i;
  slideIndex++;
  if (slideIndex > slides.length) {slideIndex = 1}
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active";
  setTimeout(showSlides, 5000); // Change image every 5 seconds
}
// -------------

window.onscroll = function() {myFunction()};
var navbar = document.getElementById("navbar");
var sticky = navbar.offsetTop;
function myFunction() {
  //console.log("pageYOffset: "+pageYOffset, "sticky: "+ sticky)
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}


function drawWeather( d ) {
    var elm_weather = document.getElementById("weather")
    var elm_description = document.getElementById('description')
	var celcius = Math.round(parseFloat(d.main.temp)-273.15);
	var fahrenheit = Math.round(((parseFloat(d.main.temp)-273.15)*1.8)+32);
	var description = d.weather[0].description;

	document.getElementById('temp').innerHTML = celcius + '&deg;';

	//alert(description)

	if( description.indexOf('rain') > 0 ) {
  	elm_weather.className = 'rainy';
	elm_description.className = 'fa fa-umbrella';
	elm_description.style.color = 'blue'

  } else if( description.indexOf('cloud') > 0 ) {
  	elm_weather.className = 'cloudy';
	elm_description.className = 'fa fa-cloud';
	elm_description.style.color = 'lightblue'

  } else if( description.indexOf('sunny') > 0 ) {
  	elm_weather.className = 'sunny';
	elm_description.className = 'fa fa-sun-o';
	elm_description.style.color = 'yellow'
  }
}

function weatherBalloon( city_name ) {
fetch('https://api.openweathermap.org/data/2.5/weather?q='+city_name+',ug&APPID=304ebb0a5c256b56de3aa522013dbe66')
  .then(response => response.json())
  .then(data => {drawWeather(data);
  });
}

function open_close_menu(is_open=true)
{
   var navbar = document.getElementById('navbar')
   var town_header = document.getElementById('town_header')
   var town_title = document.getElementById('town_title')
   var my_menu = document.getElementById('myMenu')

   var town_content = document.getElementById('town_content')
   var town_bottom = document.getElementById('town_bottom')


    if (is_open==true)
    {
       my_menu.style.width = '161px';
       navbar.style.width = '91%';
       navbar.style.marginLeft = '161px';
       town_header.style.marginLeft = '161px';
       town_header.style.width = '91%';
       town_title.style.marginLeft = '161px';
       town_title.style.width = '91%';
       town_content.style.marginLeft = '161px';
       town_content.style.width = '91%';
       town_bottom.style.marginLeft = '161px';
       town_bottom.style.width = '91%';
    } else{
       my_menu.style.width = '0px';
       navbar.style.marginLeft = '0px';
       navbar.style.width = '100%';
       town_header.style.marginLeft = '0px';
       town_header.style.width = '100%';
       town_title.style.marginLeft = '0px';
       town_title.style.width = '100%';
       town_content.style.marginLeft = '0px';
       town_content.style.width = '100%';
       town_bottom.style.marginLeft = '0px';
       town_bottom.style.width = '100%';
    }
//    alert(my_menu.outerHTML)


}

window.onload = function() {
  weatherBalloon( city_name = 'Kampala' );
}