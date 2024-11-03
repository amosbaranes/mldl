function CandlestickChartManager(tickers=null)
{
 this.tickers = tickers;
 this.container = document.getElementById("tickers_analysis");
 this.objs = {};
 for(i in this.tickers)
 {
  var t = this.tickers[i];
  var tr_ = document.createElement("tr");
  var td = document.createElement("td");td.innerHTML=t;tr_.appendChild(td);
  td = document.createElement("td");var td_div = document.createElement("div");td_div.setAttribute("id", "host_div_"+t);
  td_div.setAttribute("style", "width:500px;height=250px");
  td.appendChild(td_div);
  tr_.appendChild(td);
  td = document.createElement("td");td.setAttribute("id", "analysis_"+t);td.innerHTML=t;tr_.appendChild(td);
  this.container.appendChild(tr_);
  this.objs[t] = new CandlestickChart(ticker=t, data_function=getRandomData)
 }
}

function CandlestickChart(ticker="aapl", data_function=null)
{
 this.ticker = ticker;
 this.data_function = data_function;
 this.host_div = document.getElementById("host_div_"+this.ticker);
 this.canvas = document.createElement("canvas");
 this.canvas.setAttribute("id", "canvas_"+this.ticker);
 this.host_div.appendChild(this.canvas);
 this.ctx = this.canvas.getContext('2d');
 this.ctx.canvas.width = this.host_div.style.width;
 this.ctx.canvas.height = this.host_div.style.height;

 this.initialDateStr = '6 Jan 2022 21:28 GMT';
 this.initialDate = luxon.DateTime.fromRFC2822(this.initialDateStr);
 this.lastDate = this.initialDate;

 this.data = []
 this.label = 'Chart for '+this.ticker
 this.chart = new Chart(this.ctx, {type: 'candlestick', data: {datasets: [{label: this.label, data: this.data}]}});
 this.data_function(this);
}
CandlestickChart.prototype.add_data_point = function(d)
{
		this.data.push(d);
	    this.chart.config.data.datasets = [{label: this.label, data: this.data}]
		this.chart.update()
}

// {"IWM": ["6 Jan 2022 22:44 GMT", 219.2431, 219.41, 219.23, 219.36, 54485.0],
// "AMZN": ["6 Jan 2022 22:44 GMT", 3265.26, 3270.0, 3265.2439, 3270.0, 842.0]}

function randomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

function randomBar(date) {
    var basePrice = 30
    var open = +randomNumber(basePrice * 0.95, basePrice * 1.05).toFixed(2);
    var close = +randomNumber(open * 0.95, open * 1.05).toFixed(2);
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

function getRandomData(chart_obj, count=60) {
    var data = [randomBar(chart_obj.initialDate)];
    while (data.length < count) {
        chart_obj.lastDate = chart_obj.lastDate.plus({days: 1});
        if (chart_obj.lastDate.weekday <= 5) {
            data.push(randomBar(chart_obj.lastDate));
        }
    }
    // alert(JSON.stringify(data))
    chart_obj.data = data
    chart_obj.chart.config.data.datasets = [{label: chart_obj.label, data: chart_obj.data}]
    chart_obj.chart.update();
}
