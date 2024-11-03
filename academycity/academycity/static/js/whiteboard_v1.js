var is_in_session = "False";

// base win object
//----------------
function acWin(my_name_="none", win_name="none", win_title="none", width="500", height="300", right= "0%", top="0%",
  is_scroll=true, zindex=11)
{
  //alert(win_name +" : " + win_title +" width: " + width +" : " + height +" : " + right +" top: " + top)
  this.win_name = win_name
  this.my_name_ = my_name_
  this.win_frame = document.createElement("div");
  this.win_frame.setAttribute("id", "win_frame_"+this.win_name);
  this.style_frame = "position: fixed; background-color: white; z-index: "+zindex+"; text-align: left;"
  this.style_frame += "height:"+height+"px;width:"+width+"px;font-size:12pt;"
  this.style_frame += "border:2px solid #0094ff;"
  this.style_frame += "border-top-left-radius:35px;border-top-right-radius:35px;"
  this.style_frame += "right:"+right+";top:"+top+";"
  this.win_frame.setAttribute("style", this.style_frame);
  //alert(this.win_frame.outerHTML)

  this.title_height = 25
  this.win_frame_title = document.createElement("div");
  this.win_frame_title.setAttribute("id", "win_frame_title_"+this.win_name);
  this.style_title = "text-align:center;display:block;padding:4px;color:#fff;margin:-1 0 0 0;height:"+this.title_height+"px;z-index:11;"
  this.style_title += "background-color:#2196F3;color:#fff;cursor:move;font-size:12pt;border-top-left-radius:35px;border-top-right-radius:35px;"
  this.win_frame_title.setAttribute("style", this.style_title);
  this.win_frame_title.innerHTML = win_title
  this.win_frame.appendChild(this.win_frame_title)

  this.win_frame_ico = document.createElement("img");
  this.win_frame_ico.setAttribute("id", "win_frame_ico_"+this.win_name);
  this.win_frame_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_frame_ico.setAttribute("style", "border-radius: 10px;position: absolute;left: 75px");
  this.win_frame_ico.setAttribute("height", "20");
  this.win_frame_ico.setAttribute("width", "20");
  this.win_frame_title.appendChild(this.win_frame_ico)

  this.win_content = document.createElement("div");
  this.win_content.setAttribute("id", "win_content_"+this.win_name);
  //this.win_content.innerHTML = "Content."
  this.style_content = "height:"+(height-this.title_height-7)+"px;width:"+width+"px;"
  if(is_scroll==true)
  {
   this.style_content += "overflow: scroll;"
  }
  this.win_content.setAttribute("style", this.style_content);
  this.win_frame.appendChild(this.win_content)
  //alert(this.win_frame.outerHTML)
  //--
  this.win_nav = document.createElement("div");
  this.win_nav.setAttribute("id", "win_nav_"+this.win_name);
  this.style_win_nav = "position:fixed;text-align:center;width:10px;display:none; right:"+right+";top:"+top+";";
  this.win_nav.setAttribute("style", this.style_win_nav)

  this.win_nav_ico = document.createElement("img");
  this.win_nav_ico.setAttribute("id", "win_nav_ico_"+this.win_name);
  this.win_nav_ico.setAttribute("src", "/static/core/globs.jpg");
  this.win_nav_ico.setAttribute("style", "border-radius: 10px;position: absolute;right:97%");
  this.win_nav_ico.setAttribute("height", "20");
  this.win_nav_ico.setAttribute("width", "20");
  this.win_nav.appendChild(this.win_nav_ico)

  this.span = document.createElement("span");
  this.span.setAttribute("id", "span_"+this.win_name);
  this.span.innerHTML = "&nbsp;Whiteboard"
  this.span.setAttribute("style", "position: absolute;left: 50");
  this.win_nav.appendChild(this.span)

  var body_ = document.getElementById("body_base0")
  body_.appendChild(this.win_frame)
  body_.appendChild(this.win_nav)

  this.win_frame_title.setAttribute('pos1', 0);
  this.win_frame_title.setAttribute('pos2', 0);
  this.win_frame_title.setAttribute('pos3', 0);
  this.win_frame_title.setAttribute('pos4', 0);
}

acWin.prototype.set_acWinStat = function(ss)
{
//open and close acWin.
    if (ss != this.win_frame.style.display)
    {
      var event_close_ac = new Event("click", {bubbles: true});
      this.win_frame_ico.dispatchEvent(event_close_ac)
    }
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
  // console.log("set_acWinStatEventListeners :", "Check 0000");
  eval("var " +  ss_obj.my_name + '= ss_obj')
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
  s += 'var dic_ = {"type": "wsf", "message": my_name+","+ x_'+this.win_name+'.style.display};'
  s += 'try{'
  s += 'console.log("Check : socket.send(JSON.stringify(dic_))", dic_);'

  s += 'if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};'

  s += 'console.log("Good : socket.send(JSON.stringify(dic_))", dic_);'
  s += '} catch (err) {console.log("Error 102", err.message)}'
  s += '}.bind(elm = this.win_frame_ico, event, x_'+this.win_name+' = this.win_frame, n_'+this.win_name+' = this.win_nav, socket='+ss_obj.my_name+'.socket));'
  try{
   console.log("eval 100:"+ss_obj, s)
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
  s += 'var dic_ = {"type": "wsf", "message": my_name+","+ x_'+this.win_name+'.style.display};'
  s += 'try{'
  s += 'console.log("Check : socket.send(JSON.stringify(dic_))", dic_);'

  s += 'if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};'

  s += 'console.log("Good 1:", "socket.send(JSON.stringify(dic_))");'
  s += '} catch (err) {console.log("Error 103", err.message)}'
  s += '}.bind(this.win_nav_ico, event, x_'+this.win_name+' = this.win_frame, n_'+this.win_name+' = this.win_nav, socket='+ss_obj.my_name+'.socket));'
  try{
   //console.log("eval 200:"+ss_obj, s)
   eval(s)
   //console.log("Good :set_acWinStatEventListeners: ", "eval 200")
   } catch (err) {console.log("Error 201", err.message)}

  //console.log("set_acWinStatEventListeners :", "Check 0200");
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
  s += '}.bind(elm_win_frame_title_'+this.win_name+', event));'
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
  s += 'var dic_ = {"type": "move_frame", "message": msg_};'
  s += 'if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};'

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
  //console.log("set_acWinStatEventListeners :", "Check 0300");

  try{
    var event_frame_ico = new Event("click", {bubbles: false});
    this.win_frame_ico.dispatchEvent(event_frame_ico)
    } catch (err) {console.log("Error set_acWinStatEventListeners :", "Check 0350");}
  // console.log("set_acWinStatEventListeners :", "Check 0400");
}


acWin.prototype.remove_win = function()
{
  try{
      this.win_frame.outerHTML = ""
      this.win_navL = ""
    } catch (err) {console.log("Error remove_win :", "remove_win 0450");}
}


function openvidu(my_name_, win_name_, win_title_, socket, chat_room_name, is_instructor, use_id)
{
    this.my_name = my_name_;
    this.name = "vi_" + win_name_.toString(); // course_schedule_id
    this.width_ = "700"
    this.height_ = "500"
    var is_scroll_ = true;
    if(is_instructor){is_scroll_=true}
    acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, width=this.width_, height=this.height_, right="2%", top="30%", is_scroll=is_scroll_, zindex=20)
    this.win_frame.setAttribute("my_name", my_name_);

    this.span.innerHTML = "&nbsp;VI"
    this.socket = socket
    this.is_instructor = is_instructor
    this.user_call_id = use_id;
    console.log("Openvidu - Create my video object - My_user_call_id: ", this.user_call_id);

    this.win_frame_ico.style.right="95%";

    this.join = document.createElement("div");
    this.join.setAttribute("id", "join");
    //var h1_text = document.createElement("h1");
    //h1_text.innerHTML = 'Join a video session';
    //this.join.appendChild(h1_text);

    //this.form = document.createElement("form");

    //var pf = document.createElement("p");
    //var l = document.createElement("label");
    //l.innerHTML = 'Session:';
    //pf.appendChild(l);
    //var i = document.createElement("input");
    //i.setAttribute("type", "text");
    //i.setAttribute("id", "sessionId");
    //i.setAttribute("value", "SessionA");
    //i.setAttribute("required", "true");
    //pf.appendChild(i);
    //this.form.appendChild(pf)

    //var pf1 = document.createElement("p");
    var i1 = document.createElement("input");
    i1.setAttribute("type", "button");
    i1.setAttribute("value", "JOIN");
    //pf1.appendChild(i1);
    //this.form.appendChild(pf1);
    this.join.appendChild(i1);
    this.win_content.appendChild(this.join);
    ss = 'i1.addEventListener("click", function(event){event.preventDefault();joinSession(mySessionId = "'+this.my_name + '");})';
    try{
    eval(ss)
    } catch(er){alert(er.message)}
    //----
    this.session = document.createElement("div");
    this.session.setAttribute("id", "session");
    this.session.setAttribute("style", "display: none;");
    var h1_session = document.createElement("h1");
    h1_session.setAttribute("id", "session-header");
    this.session.appendChild(h1_session);

    var si = document.createElement("input");
    si.setAttribute("type", "button");
    si.setAttribute("value", "LEAVE");
    si.addEventListener("click", function(){ leaveSession();})
    this.session.appendChild(si);

    var div1 = document.createElement("div");
    var div11 = document.createElement("div");
    div11.setAttribute("id", "publisher");
    //var h3 = document.createElement("h3");
    //h3.innerHTML = 'YOU';
    //div11.appendChild(h3);
    div1.appendChild(div11);
    var div12 = document.createElement("div");
    div12.setAttribute("id", "subscriber");
    //var h3_ = document.createElement("h3");
    //h3_.innerHTML = 'OTHERS';
    //div12.appendChild(h3_);
    div1.appendChild(div12);
    this.session.appendChild(div1);
    this.win_content.appendChild(this.session);
    joinSession(mySessionId = this.name);
}
openvidu.prototype = Object.create(acWin.prototype)



function Video(my_name_, win_name_, win_title_, socket, chat_room_name, is_instructor, use_id)
{
    this.my_name = my_name_
    this.name = "vi_" + win_name_.toString(); // course_schedule_id
    this.width_ = "260"
    this.height_ = "330"
    var is_scroll_ = false;
    if(is_instructor){is_scroll_=true}
    acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, width=this.width_, height=this.height_, right="2%", top="15%",
     is_scroll=is_scroll_, zindex=8)
    this.win_frame.setAttribute("my_name", my_name_);

    this.span.innerHTML = "&nbsp;VI"
    this.socket = socket
    this.is_instructor = is_instructor
    this.user_call_id = use_id;  // + "_" + Math.floor(Math.random()*1000000000)

    console.log("Video - Create my video object - My_user_call_id: ", this.user_call_id);

    this.win_frame_ico.style.right="90%";
    this.allPeersConn = {};
    this.allVideosArray = {};
    //--
    //console.log("Video:", "Check 0003");

    this.myStream = null;
    this.my_video = document.createElement("video");
    this.my_video.setAttribute("id", "id_"+this.user_call_id)
    this.my_video.setAttribute("src", "")
    this.my_video.setAttribute("autoplay", "autoplay")
    this.my_video.setAttribute("width", "240")
    this.my_video.setAttribute("height", "135")
    this.win_content.appendChild(this.my_video)
    this.allVideosArray[this.user_call_id] = this.my_video

    //console.log("Video:", "Check 0004");
//
//    this.configuration = {
//        'iceServers': [{
//            'urls': 'stun:stun.l.google.com:19302'
//        }]
//    };

    this.configuration = {
        'iceServers': [
            {'urls':'stun:stun01.sipphone.com'},
            {'urls':'stun:stun.ekiga.net'},
            {'urls':'stun:stun.fwdnet.net'},
            {'urls':'stun:stun.ideasip.com'},
            {'urls':'stun:stun.iptel.org'},
            {'urls':'stun:stun.rixtelecom.se'},
            {'urls':'stun:stun.schlund.de'},
            {'urls':'stun:stun.l.google.com:19302'},
            {'urls':'stun:stun1.l.google.com:19302'},
            {'urls':'stun:stun2.l.google.com:19302'},
            {'urls':'stun:stun3.l.google.com:19302'},
            {'urls':'stun:stun4.l.google.com:19302'},
            {'urls':'stun:stunserver.org'},
            {'urls':'stun:stun.softjoys.com'},
            {'urls':'stun:stun.voiparound.com'},
            {'urls':'stun:stun.voipbuster.com'},
            {'urls':'stun:stun.voipstunt.com'},
            {'urls':'stun:stun.voxgratia.org'},
            {'urls':'stun:stun.xten.com'},
            {'urls': 'stun:stun.services.mozilla.com'},
            {
                'urls': 'turn:numb.viagenie.ca',
                'credential': 'azxcvbnm',
                'username': 'g@drbaranes.com'
            }
        ]
    };

// azxcvbnm

//var rtc = new RTCPeerConnection({
//    iceServers:[{“urls”:”stun:stun.l.google.com:19302”}]
//});


    //--
    console.log("Video - configuration:", this.configuration);
    //console.log("Video - configuration:", this.configuration);
    //console.log("End create Video - configuration:", this.configuration);
    //console.log("Video: - check", "Check 0005");
}
Video.prototype = Object.create(acWin.prototype)

Video.prototype.setMyVideo = function (vi) {
  try{
    var event_frame_ico = new Event("click", {bubbles: false});
    vi.win_frame_ico.dispatchEvent(event_frame_ico)
  } catch (err) {console.log("error in vi.win_frame_ico.dispatchEvent:", err);}

  try{
       if (navigator.mediaDevices.getUserMedia) {
            let constraints = {
                video: {width: 1280, height: 720},
                audio: true
            };
            //console.log("setMyVideo navigator.mediaDevices.getUserMedia:  ", "Check Video 0013");
            //console.log('startStream');
            navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
              vi.my_video.srcObject = stream;
              vi.myStream = stream;

              console.log("vi.myStream:  ", vi.myStream);

            }).catch(function (error) {
              console.log("Error connecting my stream:  ", error);
            });

            console.log('setMyVideo - End connecting my Stream');
       }
    } catch (err) {console.log('Error navigator.mediaDevices.getUserMedia: ' , err.message)}
    //console.log("setMyVideo:", "Check 0013");
}

Video.prototype.setConnection = function () {
    var user_call_id_ = this.user_call_id
    var socket = this.socket
    var is_instructor = this.is_instructor
    var videoSignalSocketReady = setInterval(function () {
        console.log('ready state=', socket.readyState);
        if (socket.readyState === 1) {
                    var dic_ = {'type': 'user_here', 'message': 'Are you ready for a call?',
                    'is_instructor':is_instructor, 'user_call_id': user_call_id_}
                    console.log("Video.prototype.setConnection 103:", dic_);
                    socket.send(JSON.stringify(dic_));
            clearInterval(videoSignalSocketReady);
        }
    }, 1000, user_call_id_, socket, is_instructor);
}

Video.prototype.setResponseConnection = function (msg_, vi, ss_name) {
        console.log('setResponseConnection_1: data type != user_here: ' + msg_.user_call_id);
        this.setReceiveConnection(msg_, vi, ss_name);
        var rtcPeerConn = this.allPeersConn[msg_.user_call_id]
        // sdp message means remote party made us an offer
        msg = JSON.parse(msg_.msg)
        if (msg.sdp) {
            console.log('setResponseConnection_2: data type != user_here: ' + msg_.user_call_id);
            rtcPeerConn.setRemoteDescription(
                new RTCSessionDescription(msg.sdp), function () {
                    // if we received an offer, we need to answer
                    if (rtcPeerConn.remoteDescription.type == 'offer') {
                        var sss = 'rtcPeerConn.createAnswer('+ss_name+', vi.logError);'
                        //alert(sss)
                        try{
                          eval(sss)
                        }
                        catch (err) { alert( 'err1 sss ' + sss)}
                    }
                }, vi.logError);
        } else {
            console.log('setResponseConnection_3 data type != user_here, msg_.msg.candidate: ' + msg_.user_call_id +" : " + msg.candidate.candidate);
            rtcPeerConn.addIceCandidate(new RTCIceCandidate(msg.candidate));
        }
}


var __n = 0;
Video.prototype.setReceiveConnection = function (msg_, vi, ss_name) {
  //alert(msg_.is_instructor)
   if (msg_.is_instructor == "True" || this.is_instructor == "True")
   {
     //alert(msg_.type)
     console.log("setReceiveConnection1 msg_.type :  ", msg_.type)
     var rtcPeerConn = this.allPeersConn[msg_.user_call_id]
     if(rtcPeerConn == null )
     {
        //alert(msg_.user_call_id)
        rtcPeerConn = new RTCPeerConnection(this.configuration);
        //alert(rtcPeerConn)
        this.allPeersConn[msg_.user_call_id] = rtcPeerConn
        //alert(this.allPeersConn[msg_.user_call_id])
        //console.log("setReceiveConnection2:  ", msg_.type)

        var video = this.allVideosArray[msg_.user_call_id];
        if(video == null )
        {
         video = document.createElement("video");
         video.setAttribute("src", "")
         video.autoplay = true
         video.id = "id_" + msg_.user_call_id
         video.width="240"
         video.height="135"
         video.setAttribute("autoplay", "autoplay")
         //alert(video.outerHTML)
         this.win_content.appendChild(video);
         this.allVideosArray[msg_.user_call_id] = video
        }
        //alert(this.win_content.outerHTML)
        //console.log("setReceiveConnection3:  ", msg_.type)

        rtcPeerConn.onicecandidate = function (evt) {
            if (evt.candidate) {

                console.log("rtcPeerConn.onicecandidate: evt.candidate: " + evt.candidate)

                vi.socket.send(JSON.stringify({
                    'type': 'ice candidate',
                    'message': JSON.stringify({
                        'candidate': evt.candidate
                    }),
                    'is_instructor': vi.is_instructor,
                    'user_call_id': vi.user_call_id
                }));
            }
        }

        ss = 'rtcPeerConn.onnegotiationneeded = function () {'

        ss += 'console.log("rtcPeerConn.onnegotiationneeded to create offer: 1 ");'

        ss += 'vi.allPeersConn["'+msg_.user_call_id+'"].createOffer('+ss_name+', vi.logError);'
        ss += '}'
        eval(ss)
        //alert(ss)
        //console.log("setReceiveConnection5:  ", ss)



     // localStream.getTracks().forEach(track => myPeerConnection.addTrack(track, localStream));



        ss = 'rtcPeerConn.onaddstream = function (event) {'

        ss += 'console.log("rtcPeerConn.onaddstream_0 __n: ", parseInt('+msg_.user_call_id+') != parseInt(vi.user_call_id));'

        //ss += 'console.log("rtcPeerConn.onaddstream_0  vi.user_call_id: ", vi.allVideosArray["'+vi.user_call_id+'"].outerHTML);'
        //ss += 'console.log("rtcPeerConn.onaddstream_0  msg_.user_call_id: ", vi.allVideosArray["'+msg_.user_call_id+'"].outerHTML);'

        ss += 'if (parseInt('+msg_.user_call_id+') != parseInt(vi.user_call_id)){'


        ss += 'try{'

        ss += 'console.log("event.stream of other side:  ", event.stream);'

        //ss += 'console.log("----before setting stream------0");'

        //ss += 'console.log("rtcPeerConn.onaddstream_0 msg_.user_call_id: ", vi.allVideosArray["'+msg_.user_call_id+'"].outerHTML);'

        ss += 'vi.allVideosArray["'+msg_.user_call_id+'"].srcObject = event.stream;'


        ss += 'console.log("----After setting stream------1");'

        ss += 'console.log("rtcPeerConn.onaddstream_0 msg_.user_call_id: ", vi.allVideosArray["'+msg_.user_call_id+'"].outerHTML);'

        ss += 'console.log("----After setting stream------1");'

        ss += '} catch (err) {console.log(err.message)} }'

        ss += 'console.log("rtcPeerConn.onaddstream_1");'

        ss += '}'
        eval(ss)


        // alert(ss)
        rtcPeerConn.addStream(this.myStream); // this triggers event our peer needs to get our stream
        //alert(this.win_content.outerHTML)
        console.log("setReceiveConnectionEnd:  ", msg_.type)
     }
   }
}


Video.prototype.sendLocalDesc = function (desc, user_call_id, vi) {
 var rtcPeerConn = this.allPeersConn[user_call_id]
    console.log("sendLocalDesc by user_call_id : ", user_call_id)
    rtcPeerConn.setLocalDescription(desc, function () {
        vi.socket.send(JSON.stringify({
            'type': 'SDP',
            'message': JSON.stringify({
                'sdp': rtcPeerConn.localDescription
            }),
            'is_instructor': vi.is_instructor,
            'user_call_id': user_call_id
        }));
    }, vi.logError);
        console.log("sendLocalDescEnd", user_call_id)
}

Video.prototype.logError = function (error) {
    console.log(error.name, error.message);
}


// Chat
//----------------
function Chat(my_name_, win_name_, win_title_, socket, chat_room_name)
{
  this.name = "ct_" + win_name_.toString(); // course_schedule_id
  this.width_ = "400"
  this.height_ = "300"
  acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, width=this.width_, height=this.height_, right="2%", top="20%",
   is_scroll=true, zindex=9)
  this.my_name = my_name_
  this.win_frame.setAttribute("my_name", my_name_);

  this.span.innerHTML = "&nbsp;CT"
  this.socket = socket
  this.win_frame_ico.style.right="90%";

  this.input_msg_container = document.createElement("div");
  this.input_msg_container.setAttribute("class", "container")
  // alert("input_msg_container_"+this.name)

  this.input_msg_container.setAttribute("id", "input_msg_container_"+this.name);
  // this.input_msg_container.innerHTML = "<br/><br/>"
  this.input_msg = document.createElement("input");
  this.input_msg.setAttribute("id", "input_msg_"+this.name)
  this.input_msg.setAttribute("style", "padding:0px; width:"+(this.width_-30)+"px")
  this.input_msg.setAttribute("value", "")
  this.input_msg_container.appendChild(this.input_msg)
  this.win_content.appendChild(this.input_msg_container)
  this.chat_items = document.createElement('ul');
  this.chat_items.setAttribute('id','chat_items_'+this.name);
  this.win_content.appendChild(this.chat_items)
}
Chat.prototype = Object.create(acWin.prototype)

Chat.prototype.setChatMessage = function(msg_)
{
  this.chat_items.innerHTML = "<li>" + msg_.username + ": " + msg_.msg + "</li>" + this.chat_items.innerHTML
}

Chat.prototype.setOnSocketOpen = function()
{
  this.input_msg.addEventListener("keyup", function(){
        if (event.keyCode === 13) {  // enter, return
            var val = this.value
            var dic_ = {'type': 'chat_message', 'message': val}
            try{
                 //alert(dic_.type +" : " +dic_.message)
                 socket.send(JSON.stringify(dic_))
                 //if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};
            } catch (err) {alert(err.message)}
            this.value = ""
        }
  }.bind(input_msg = this.input_msg, event, socket=this.socket));
}


// Whiteboard
//----------------
function Whiteboard (my_name_, win_name_, win_title_, socket, chat_room_name, link_get_wb, link_get_wb_of_section, use_id, is_instructor)
{
  this.name = "wb_" + win_name_.toString(); // course_schedule_id
  this.course_schedule_id = win_name_
  this.link_get_wb = link_get_wb
  this.link_get_wb_of_section = link_get_wb_of_section
  this.use_id = use_id
  this.is_instructor = is_instructor
  this.section_id = -1
  this.width_ = "1000"
  this.height_ = "500"
  acWin.call(this, my_name_=my_name_, win_name=this.name, win_title=win_title_, width=this.width_, height=this.height_, right="2%", top="40%", is_scroll=false)
  this.span.innerHTML = "&nbsp;WB"
  this.socket = socket
  this.my_name = my_name_
  this.whiteboardsData = {}
  this.win_frame.setAttribute("my_name", my_name_);

    this.colors = ['black', 'green', 'red', 'blue', 'yellow', 'white']
    this.color_match = {}
    this.color_match['#000000']= 'black';
    this.color_match['#008000']= 'green';
    this.color_match['#ff0000']= 'red';
    this.color_match['#0000ff']= 'blue';
    this.color_match['#ffff00']= 'yellow';
    this.color_match['#ffffff']= 'white';


    //--
//alert(this.win_frame.outerHTML)
    this.drawing_colors = document.createElement("div");
    this.drawing_colors.setAttribute("id", "drawing_colors"+this.name)
    this.win_content.appendChild(this.drawing_colors)
    //--
//alert(this.win_frame.outerHTML)
    this.tab_titles = document.createElement("ul");
    this.tab_titles.setAttribute("id", "titles-" + this.name);
    this.tab_titles.setAttribute("class", "nav nav-tabs")
    this.win_content.appendChild(this.tab_titles)
    //--
    this.board_contents = document.createElement("div");
    this.board_contents.setAttribute("id", "contents-" + this.name);
    this.board_contents.setAttribute("class", "tab-content")
    this.board_contents.setAttribute("style", "overflow: scroll;")
    this.win_content.appendChild(this.board_contents)
    this.TabObjects = {}
    //--
    this.button_add = document.createElement("button");
    this.button_add.setAttribute("id", "button_add")
    //this.button_add.setAttribute("class", "button")
    this.button_add.setAttribute("style", "background-color: lightblue;margin: 10px 2px;padding: 10px;border-radius: 20%;")
    //border-radius: 50%;padding: 1px;margin: 1px 2px;
    //--
    this.button_add.innerHTML = 'Add Whiteboard'
    this.button_add.addEventListener("click", function(){
         console.log("button_add Whiteboard 100:")
         event.preventDefault();

         if(is_in_session == "False"){return;}

         var ss = prompt('Enter Whiteboard name:', '')
         if(ss != "" && ss != null)
         {
            console.log("button_add board_name 100:", ss)
            console.log("wm.link_get_wb 110:", wm.link_get_wb)
            console.log("wm.section_id 120:", wm.section_id)
            $.post(wm.link_get_wb,
            {
              wb_id: "-1",
              section_id: wm.section_id,
              board_name: ss,
              course_schedule_id : win_name_
            },
            function(data){
                try{
                    //--
                    //wm.n_whiteboard = data['wb_id']

                    console.log("button_add 101:", "wb_id")
                    n_wb_id = data['wb_id']
                    console.log("button_add 102:", n_wb_id)
                    //--
                    wm.setTab(n_wb_id, ss, is_send_tab_creation="1")
                    wm.setDefaultColors("black")
                    wm.pen_eraser.style.display = "inline"
                    wm.pen_eraser_size.display = "inline"

                    var dic_ = {'type': 'add_white_board', 'message': n_wb_id+","+ss}
                    //console.log("button_add 103:", dic_)
                    socket.send(JSON.stringify(dic_));
                    //console.log("button_add 104:", "after socket - 104")
                } catch (e) {alert(e)}
            });
         }
      }.bind(cc=this.button_add, event, socket=this.socket, wm=this));
    this.drawing_colors.appendChild(this.button_add)
    //--
    for(var key in this.color_match)
    {
      var c = this.color_match[key];
        var cc1 = document.createElement("button");
        cc1.innerHTML = c.charAt(0).toUpperCase()
        cc1.setAttribute("id", "id_"+c+"_"+this.name)
        // alert("id_"+c+"_"+wm.name)
        //cc1.setAttribute("class", "button")
        cc1.setAttribute("style", "color:black;border-radius: 40%;padding: 8px;margin: 4px 4px; background-color: "+c)
        cc1.addEventListener("click", function(){
          event.preventDefault();
          cc1 = event.target
          event.preventDefault();
          wm.CurrentWhiteboardTab.ctx.strokeStyle = cc1.style.backgroundColor
          ss = wm.setDefaultColors(cc1.style.backgroundColor)
          var dic_ = {'type': 'wc', 'message': ss}

          // socket.send(JSON.stringify(dic_))

          if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};

        }.bind(cc1, event, socket=this.socket, wm=this));
        //alert(cc.outerHTML)
        this.drawing_colors.appendChild(cc1)
    }

    //-- mode: pen or eraser --
    this.pen_eraser = document.createElement("button");
    // this.pen_eraser.setAttribute("class", "button")
    this.pen_eraser.setAttribute("id", "id_pen_eraser_" + this.name)

    this.pen_eraser.setAttribute("style", "height:40px;padding:8px;margin: 5px 5px;background-color:red;border-radius:15px;display:inline")
    this.pen_eraser.innerHTML = 'Eraser'
    this.pen_eraser.addEventListener("click", function(){
        event.preventDefault();
        var mode = pen_eraser.innerHTML
        if(mode=="Pen"){
          pen_eraser.innerHTML = "Eraser"
          pen_eraser.style.backgroundColor = "green"
          wm.CurrentWhiteboardTab.elm_canvas.setAttribute("mode", "1");mode=1
        }else{
          pen_eraser.innerHTML = "Pen";
          pen_eraser.style.backgroundColor = "red";
          wm.CurrentWhiteboardTab.elm_canvas.setAttribute("mode", "0"); mode=0;
        }
        wm.setPenEraser(wm.CurrentWhiteboardTab.elm_canvas.getAttribute("mode"));
        var dic_ = {'type': 'wp', 'message': mode}
        // socket.send(JSON.stringify(dic_))

        if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};

    }.bind(pen_eraser=this.pen_eraser, event, wm=this, socket=this.socket));
    this.drawing_colors.appendChild(pen_eraser)

  this.pen_eraser_size = document.createElement("INPUT");
  this.pen_eraser_size.setAttribute("type", "text");
  this.pen_eraser_size.setAttribute("value", "2");
  this.pen_eraser_size.setAttribute("style", "display:inline;width:35px");
  this.pen_eraser_size.addEventListener("change", function(){
        event.preventDefault();
        val = event.target.value
        var dic_ = {'type': 'wfontsize', 'message': val}
        //socket.send(JSON.stringify(dic_))

        if(is_in_session == "True"){socket.send(JSON.stringify(dic_));};

    }.bind(pen_eraser_size=this.pen_eraser_size, event, wm=this, socket = this.socket));

  this.drawing_colors.appendChild(this.pen_eraser_size)
  var event_close_wb = new Event("click", {bubbles: false});
  this.win_frame_ico.dispatchEvent(event_close_wb)
}
Whiteboard.prototype = Object.create(acWin.prototype)

Whiteboard.prototype.getColorIndex = function(color)
{
    return this.colors.indexOf(color)
}

Whiteboard.prototype.setDefaultColors = function(cc)
{
    if(cc.indexOf("#") >=0){cc = this.color_match[cc]}
    for (key in this.color_match)
    {
        var c = this.color_match[key]
        try{
        document.getElementById("id_"+c+"_"+this.name).style.borderRadius = "40%"
        } catch(err) {
        alert(err.message)
        }
    }
    try{
    document.getElementById("id_"+cc+"_"+this.name).style.borderRadius = "25%"
    }  catch(err1) {
        alert(err1.message)
        }
    return cc;
}

Whiteboard.prototype.setPenEraser = function(mode)
{
  if(mode=="1")
  {
    this.pen_eraser.innerHTML = "Eraser"
    this.pen_eraser.style.backgroundColor = "green"
    //this.pen_eraser_size.style.display = "none"
  } else
  {
    this.pen_eraser.innerHTML = "Pen"
    this.pen_eraser.style.backgroundColor = "red"
    //this.pen_eraser_size.style.display = "inline"
  }
}

Whiteboard.prototype.setWBPenEraserStat = function(mode)
{
    if (mode != this.CurrentWhiteboardTab.elm_canvas.getAttribute("mode"))
    {
      var event_pen_eraser = new Event("click", {bubbles: false});
      this.pen_eraser.dispatchEvent(event_pen_eraser)
    }
}

Whiteboard.prototype.setTab = function(n_wb_id, title, is_send_tab_creation="1") {
     this.TabObjects[n_wb_id] = new WhiteboardTab(this, n_wb_id, title, width=this.width_, height=this.height_)
     this.TabObjects[n_wb_id].tabTitle.setAttribute("is_send_tab_creation", is_send_tab_creation)
     var event_set_tab = new Event("click", {bubbles: true});
     this.TabObjects[n_wb_id].tabTitle.dispatchEvent(event_set_tab)
     this.TabObjects[n_wb_id].tabTitle.setAttribute("is_send_tab_creation", 1)
     //alert(this.elm_whiteboard.outerHTML)
}

// Each Tab is a whiteboard
//-------------------------
function WhiteboardTab(parent, n_wb_id, title, width="1000", height="500", top="15", right="25", background_color="#f1f1f1",
 background_color_header="#2196F3", border_color="#f1f1f1", header_color="#fff", z_index=10)
{
  this.name = n_wb_id;
  this.parent=parent; this.title=title;
  this.width=width; this.height=height;
  this.background_color=background_color; this.background_color_header=background_color_header;
  this.border_color=border_color; this.header_color=header_color;this.z_index=z_index;
  var ed_id = "wb_" + this.parent.name + "_" + this.name
  this.tabDoc = document.createElement("div");
  this.tabDoc.setAttribute("id", ed_id)
  // need to fix for language in Arabic
  var s_style = "background-color:"+this.background_color+";z-index:"+this.z_index
  s_style += ";text-align:left;border: 3px solid "+this.border_color+";width:"+(this.width-14)+"px; height:"+(this.height-130)+"px;"
  this.tabDoc.setAttribute("style", s_style)
  this.tabDoc.setAttribute("class", "tab-pane fade ")
  this.parent.board_contents.appendChild(this.tabDoc)
  //--
  this.elm_canvas = document.createElement("canvas");
  this.elm_canvas.setAttribute("id",this.parent.name+"_"+this.name+"_canvas")
  this.elm_canvas.setAttribute("width", (this.width-25))
  this.elm_canvas.setAttribute("height", (this.height-144))
  this.elm_canvas.setAttribute("class", "pencil")
  this.elm_canvas.setAttribute("is_drawing", "false")
  this.elm_canvas.setAttribute("mode", "1")
  // pen = 1  eraser = 0
  this.elm_canvas.setAttribute("style", "border:3px solid #000000")
  // could be 3d, if you want to make a video game
  this.ctx = this.elm_canvas.getContext('2d');
  this.ctx.lineJoin = 'round'; this.ctx.lineCap = 'round'; this.ctx.lineWidth = 1;
  this.ctx.strokeStyle = 'black';
  this.elm_canvas.setAttribute("lastX", 0)
  this.elm_canvas.setAttribute("lastY", 0)
  //--
  this.elm_canvas.setAttribute("obj_name", this.name)
  //--
  this.elm_canvas.addEventListener('mouseup', (e) => { e.target.setAttribute("is_drawing", "false");});
  this.elm_canvas.addEventListener('mouseout', (e) => {e.target.setAttribute("is_drawing", "false");});
  this.elm_canvas.addEventListener('mousedown', (e) => {
    var elm = e.target; elm.setAttribute("is_drawing", "true");
    elm.setAttribute("lastX", e.offsetX);
    elm.setAttribute("lastY", e.offsetY);
  });

  this.elm_canvas.addEventListener("mousemove", function(){
    // stop the function if they are not mouse down
    var elm = event.target
    var is_drawing = elm.getAttribute("is_drawing")
    if(is_drawing == "false") return;
    //--
    obj_name = elm.getAttribute("obj_name")
    var ctx = parent.TabObjects[obj_name].ctx
    //--
    ctx.beginPath();
    var lastX = elm.getAttribute("lastX")
    var lastY = elm.getAttribute("lastY")
    //--
    var mode = elm.getAttribute("mode")
    var pen_eraser_size = parent.pen_eraser_size.value;
    if(mode=="1"){
      //pen_size = parent.pen_size.value;
      ctx.globalCompositeOperation="source-over";
      ctx.lineWidth = pen_eraser_size;
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(event.offsetX, event.offsetY);
      ctx.stroke();
    }else{
      ctx.globalCompositeOperation="destination-out";
      ctx.arc(lastX,lastY,pen_eraser_size,0,Math.PI*2,false);
      ctx.fill();
    }
    // --
    elm.setAttribute("lastX", event.offsetX)
    elm.setAttribute("lastY", event.offsetY)
    var co = parent.color_match[parent.CurrentWhiteboardTab.ctx.strokeStyle]
    co = parent.getColorIndex(co)
    var ss = co + "," + lastX + "," + lastY + "," + event.offsetX + "," + event.offsetY + "," + mode + "," + pen_eraser_size
    ss += "," + obj_name +","+parent.use_id
     var dic_ = {'type': 'wm', 'message': ss}
     // parent.socket.send(JSON.stringify(dic_))

     if(is_in_session == "True"){parent.socket.send(JSON.stringify(dic_));};

  }.bind(elm_canvas = this.elm_canvas, event, parent));

  //, obj_name=name
  //--
  this.tabDoc.appendChild(this.elm_canvas)
  //--
  var e = document.createElement("li")
  this.tabTitle = document.createElement("a")
  this.tabTitle.setAttribute("id", "tab-" + this.name)
  this.tabTitle.setAttribute("obj_name", this.name)
  this.tabTitle.setAttribute("class", "badge badge-pill badge-primary")
  this.tabTitle.setAttribute("style", "font-size:20px")
  this.tabTitle.setAttribute("href", "#" + ed_id)
  this.tabTitle.innerHTML = title
  e.appendChild(this.tabTitle)
  //--
//  var span_save = document.createElement("span")
//  span_save.setAttribute("class", "tooltip_ badge badge-pill badge-secondary")
//  var style_span_save = "cursor: context-menu;"
//  span_save.setAttribute("style", style_span_save);
//  span_save.innerHTML = "S"
//  var span_save_tip = document.createElement("span")
//  span_save_tip.setAttribute("class", "tooltiptext")
//  span_save_tip.innerHTML = "save this whiteboard"
//  span_save.appendChild(span_save_tip)
  //--
//  span_save.addEventListener("click", function(){
//    event.preventDefault();
//    alert(2222)
//  }.bind(span_save, event, parent));
//  e.appendChild(span_save)

//  var span_delete = document.createElement("span")
//  span_delete.setAttribute("class", "tooltip_ badge badge-pill badge-warning")
//  var style_span_delete = "cursor: context-menu;"
//  span_delete.setAttribute("style", style_span_delete);
//  span_delete.innerHTML = "D"

//  var span_delete_tip = document.createElement("span")
//  span_delete_tip.setAttribute("class", "tooltiptext")
//  span_delete_tip.innerHTML = "Delete this whiteboard"
//  span_delete.appendChild(span_delete_tip)

//  span_delete.addEventListener("click", function(){
//    event.preventDefault();
//    alert(3333)
//  }.bind(span_delete, event, parent));
//  e.appendChild(span_delete)

  this.parent.tab_titles.insertBefore(e, this.parent.tab_titles.childNodes[0])
  this.tabTitle.addEventListener("click", function(){
    event.preventDefault();
    var elm = event.target; var obj_name = elm.getAttribute('obj_name');
    var obj_id = elm.getAttribute('id');
    //alert('click - elm - ' +elm.outerHTML)
    parent.CurrentWhiteboardTab = parent.TabObjects[obj_name]
    try {
      $(this).tab('show');    //--
    parent.setDefaultColors(parent.CurrentWhiteboardTab.ctx.strokeStyle);
    //--
    parent.setPenEraser(parent.CurrentWhiteboardTab.elm_canvas.getAttribute("mode"));
    //--
    is_send_tab_creation = elm.getAttribute("is_send_tab_creation")
    //alert("is_send_tab_creation " + is_send_tab_creation)
    if(is_send_tab_creation != "0")
    {
      //alert('100 send tab message: '+ this.name)
      var dic_ = {'type': 'wt', 'message': "" + obj_id}
      //parent.socket.send(JSON.stringify(dic_))

      if(is_in_session == "True"){parent.socket.send(JSON.stringify(dic_));};
    }
    }
    catch(err) {alert(err.message);}
  }.bind(this.tabTitle, event, parent));
 //this.tabTitle = ea; // document.getElementById("tab-"+this.name)
 //this.tabDoc = ed; //document.getElementById(this.name)

}

Whiteboard.prototype.setSectionId = function(section_id)
{
 if(this.section_id != section_id)
 {
   var wbd = this.whiteboardsData[section_id]
   this.tab_titles.innerHTML="";
   this.board_contents.innerHTML="";
   this.TabObjects = {};
   //alert(wbd)
   if(wbd == null)
   {
        $.post(wm.link_get_wb_of_section,
        {
          section_id: section_id,
          course_schedule_id: this.course_schedule_id
        },
        function(data){
            n = 0
            for (k in data)
            {n += 1; break; }
            if(n > 0)
            {
                wm.whiteboardsData[section_id] = data;
                //alert(data)
                wm.setSectionId_update_screens(wbd = data)
            }
        });
   }
   else
   {
    //alert(wbd)
    wm.setSectionId_update_screens(wbd)
   }
   this.section_id = section_id
 }
}

Whiteboard.prototype.setSectionId_update_screens = function(wbd)
{
    for (board_id in wbd)
    {
      wm.duplicate_add_whiteboard(board_id, wbd[board_id]['board_name'])
      for(h in wbd[board_id]['data'])
      {
        var co = wbd[board_id]['data'][h]['color']
        var xf = wbd[board_id]['data'][h]['xf']
        var yf = wbd[board_id]['data'][h]['yf']
        var xt = wbd[board_id]['data'][h]['xt']
        var yt = wbd[board_id]['data'][h]['yt']
        var mode = wbd[board_id]['data'][h]['mode']
        var pen_size = wbd[board_id]['data'][h]['pen_size']
        var ss = co + "," + xf + "," + yf + "," + xt + "," + yt + "," + mode + "," + pen_size
        wm.duplicate_onmove(ss)
      }
    }
}

Whiteboard.prototype.duplicate_add_whiteboard = function(n, ss)
{
           wm.setTab(n, ss, is_send_tab_creation="0")
           wm.setDefaultColors("black")
}

Whiteboard.prototype.duplicate_onmove = function(ss)
{
    var ll=ss.split(",");
    if (ll[5]=="true"){ll[5]=1}
    var ctx = this.CurrentWhiteboardTab.ctx
    ll[0] = this.colors[ll[0]]
    ctx.strokeStyle = ll[0]
    ctx.beginPath();
    if(ll[5]=="1"){
      //pen_size = parent.pen_size.value;
      ctx.globalCompositeOperation="source-over";
      ctx.lineWidth = ll[6];
      ctx.moveTo(ll[1], ll[2]);
      ctx.lineTo(ll[3], ll[4]);
      ctx.stroke();
    }else{
      ctx.globalCompositeOperation="destination-out";
      ctx.arc(ll[1],ll[2],ll[6],0,Math.PI*2,false);
      ctx.fill();
    }
 }

