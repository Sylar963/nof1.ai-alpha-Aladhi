// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=6
indicator("Opening Range with Breakouts & Targets [LuxAlgo]",shorttitle = "LuxAlgo - ORB & Targets", overlay = true, max_lines_count = 500, max_labels_count = 500, max_boxes_count = 500)

//---------------------------------------------------------------------------------------------------------------------}
//inputs
//---------------------------------------------------------------------------------------------------------------------{

//Historical Display
showHist = input.bool(true, title = "Show Historical Data", group = "Historical Display", tooltip = "Displays All Data from Previous Sessions")
//Opening Range
orTF = input.timeframe("30", title = "Time Period", group = "Opening Range", tooltip = "Sets the Length of Time used for determining Opening Range.") 
//Custom Range
crTog = input.bool(false, title = "", inline = "Custom", group = "Custom Range")
crSesh = input.session("0930-0945", title = "", inline = "Custom", group = "Custom Range")
tz = input.string("UTC-5", title = "", group = "Custom Range", inline = "Custom", options = ["UTC-10", "UTC-8", "UTC-7", "UTC-6", "UTC-5", "UTC-4", "UTC-3", "UTC+0", "UTC+1", "UTC+2", "UTC+3", "UTC+3:30", "UTC+4", "UTC+5", "UTC+5:30", "UTC+5:45", "UTC+6","UTC+6:30", "UTC+7", "UTC+8", "UTC+9", "UTC+9:30", "UTC+10", "UTC+11", "UTC+12", "UTC+12:45", "UTC+13"])
//Breakout Signals
sigTog = input.bool(true, title = "Show Breakout Signals", group = "Breakout Signals")
useBias = input.string("No Bias", title = "Signal Bias", options = ["No Bias","Daily Bias"], group = "Breakout Signals", tooltip = "OR Fill Color is directional based on if the current Day/Session ORM is Above or Below the Previous ORM.\nExamples\nNo Bias: Signals Occur Regardless of OR Color\nDaily Bias: Signals do not fire until Target 1 when Breakout is in opposite direction of OR Color.") == "No Bias" ? false : true
sigSize = str.lower(input.string("Small", title = "Signal Size", options = ["Tiny","Small","Normal","Large","Huge"], group = "Breakout Signals"))
upSigColor = input.color(#089981, title = "Up Color", group = "Breakout Signals", inline = "Colors")
downSigColor = input.color(#f23645, title = "Down Color", group = "Breakout Signals", inline = "Colors")
//Targets
tTog = input.bool(true, title = "Show Targets", group = "Targets")
// MODIFIED: Changed to target % from mid-range
tMidPer = input.float(1.89, minval = 0.1, title = "Target % from Mid-Range", group = "Targets", tooltip = "Sets the first target as a specific percentage from the mid-range (ORM).")*0.01
// NEW: Input for subsequent targets spacing
tSubsequentPer = input.float(50, minval = 1, title = "Subsequent Targets % of Range", group = "Targets", tooltip = "Uses this % of OR Width for spacing between subsequent targets after the first one.")*0.01
tSrc = input.string("Close", title = "Target Cross Source", options = ["Close","Highs/Lows"], group = "Targets", tooltip = "Uses this Source to tell the script when a target is hit in order to draw the next target.")
tDispType = input.string("Adaptive", title = "Target Display", options = ["Adaptive","Extended"], group = "Targets", tooltip = "Adaptive: Displays and hides targets Adaptivly based on the current price.\nExtended: Extends all targets to the current bar and does not hide any targets after generation.")
//Session Moving Average
maTog = input.bool(false, title = "", inline = "MA", group = "Session Moving Average")
maLen = input.int(20, title = "", inline = "MA", group = "Session Moving Average")
maType = input.string("EMA", title = "", options=["SMA", "EMA", "RMA", "WMA", "VWMA"], inline = "MA", group = "Session Moving Average")
maColor = input.color(color.orange, title = "", inline = "MA", group = "Session Moving Average", tooltip  = "Moving average resets on the start of each session (at Opening Range Start).")
//Style
green = input.color(color.new(#089981,60), title = " Bull Target Color", group = "Style", inline = "Bull")
red = input.color(color.new(#f23645,60), title = "Bear Target Color", group = "Style", inline = "Bear")
greenFill = input.color(color.new(#089981,80), title = " Bull Fill Color", group = "Style", inline = "Bull")
redFill = input.color(color.new(#f23645,80), title = "Bear Fill Color", group = "Style", inline = "Bear")
orColor = input.color(#787b86, title = "   OR Levels Color", group = "Style", inline = "Range")
orFillColor = input.color(color.new(color.gray,60), title = "OR Highlight Color", group = "Style", inline = "Range")
tStyle = input.string("___", title = "Target Style", options = ["___","- - -",". . ."], group = "Style")
txtSize = str.lower(input.string("Small", title = "Text Size", options = ["Tiny","Small","Normal","Large","Huge"], group = "Style"))

invis = color.rgb(0,0,0,100)
//---------------------------------------------------------------------------------------------------------------------}
//Error
//---------------------------------------------------------------------------------------------------------------------{
if timeframe.in_seconds(timeframe.period) >= timeframe.in_seconds("D")
    runtime.error("Timeframe is too High! Please Reduce Timeframe to be Less-Than 1 Day.")

//---------------------------------------------------------------------------------------------------------------------}
//Functions
//---------------------------------------------------------------------------------------------------------------------{
fz(_val) => _val == 0 ? 1 : _val

day_ma(_start,_type,s,l) =>
    bs_nd = fz(ta.barssince(_start))
    v_len = bs_nd < l?bs_nd:l
    var float ma = na
    if _type == "EMA"
        k = 2/(v_len + 1)
        ma := (s*k) + (nz(ma[1])*(1-k))
    if _type == "RMA"
        a = (1/v_len)
        ma := a * s + (1 - a) * nz(ma[1])
    if _type == "SMA"
        ma := ta.sma(s,v_len)
    if _type == "WMA"
        ma := ta.wma(s,v_len)    
    if _type == "VWMA"
        ma := ta.vwma(s,v_len)
    ma

dash() => (bar_index/2 - math.floor(bar_index/2)) > 0

linestyle(_input) =>
    _input == "___"?line.style_solid:
     _input == "- - -"?line.style_dashed:
     _input == ". . ."?line.style_dotted:
     na

get_1up(_val) => (_val - math.floor(_val)) > 0 ? int(math.floor(_val) + 1) : int(_val)

//---------------------------------------------------------------------------------------------------------------------}
//UDTs
//---------------------------------------------------------------------------------------------------------------------{

type target
    line ln
    label lab

//---------------------------------------------------------------------------------------------------------------------}
//Variables
//---------------------------------------------------------------------------------------------------------------------{

var up_targs = array.new<target>(na)
var down_targs = array.new<target>(na)
var signals = array.new_label()

var bool or_sesh = false
var bool or_token = false

var float orh = na 
var float orl = na
var float hst = na 
var float lst = na
var float prev_orm = na

var int up_count = na
var int down_count = na

var box or_bx = na

var int day_dir = 0
var down_check = false
var up_check = false
down_signal = false
up_signal = false

var target h_ln = target.new(na,na)
var target l_ln = target.new(na,na)
var line m_ln = na

// NEW: Variables for mid-range based targets
var float first_up_target = na
var float first_down_target = na
var bool first_up_hit = false
var bool first_down_hit = false

//---------------------------------------------------------------------------------------------------------------------}
//Calculations
//---------------------------------------------------------------------------------------------------------------------{

orm = math.avg(orh,orl)
orw = math.abs(orh-orl)    

h_src = tSrc == "Close" ? close : high
l_src = tSrc == "Close" ? close : low

//Establishing When the OR Sesssion is

new_tf = timeframe.change(orTF)
if crTog
    or_sesh := not na(time(timeframe.period, crSesh, tz))
else 
    if session.isfirstbar
        or_sesh := true
    else if not session.isfirstbar and new_tf
        or_sesh := false

or_start = or_sesh and not or_sesh[1]
or_end = or_sesh[1] and not or_sesh

//On Start of OR Session
if or_start
    for targ in up_targs
        if showHist == false
            targ.ln.delete()
        targ.lab.delete()

    for targ in down_targs
        if showHist == false
            targ.ln.delete()
        targ.lab.delete()

    for lab in signals
        if showHist == false
            lab.delete()

    if showHist == false
        or_bx.delete()
        h_ln.ln.delete()
        l_ln.ln.delete()
        m_ln.delete()

    h_ln.lab.delete()
    l_ln.lab.delete()
    up_targs.clear()
    down_targs.clear()

    orh := high
    orl := low
    prev_orm := orm[1]
    up_count := 0
    down_count := 0
    up_check := true
    down_check := true
    first_up_hit := false
    first_down_hit := false

    or_bx := box.new(bar_index,high,bar_index,low, bgcolor = orFillColor, border_width = 0)

    or_token := false

//Running while OR Session is Live
if or_sesh
    if high > orh
        orh := high
    if low < orl
        orl := low
    or_bx.set_top(orh)
    or_bx.set_bottom(orl)
    or_bx.set_right(bar_index)
    if orh != orl
        or_token := true
        
//On End of OR Session
if or_end and or_token
    h_ln := target.new(line.new(bar_index,orh,bar_index,orh, color = orColor),label.new(bar_index,orh,text = "ORH",tooltip = str.tostring(orh,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(orColor,0), size = txtSize))
    l_ln := target.new(line.new(bar_index,orl,bar_index,orl, color = orColor),label.new(bar_index,orl,text = "ORL",tooltip = str.tostring(orl,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(orColor,0), size = txtSize))
    m_ln := line.new(bar_index,orm,bar_index,orm, style = line.style_dashed, color = orColor)
    
    // MODIFIED: Calculate first targets based on mid-range percentage
    first_up_target := orm + (orm * tMidPer)
    first_down_target := orm - (orm * tMidPer)
    
    // Initialize subsequent targets based on OR width
    hst := first_up_target + (orw*tSubsequentPer)
    lst := first_down_target - (orw*tSubsequentPer)

    day_dir := orm > prev_orm ? 1 : orm < prev_orm ? -1 : 0
    linefill.new(h_ln.ln,l_ln.ln, day_dir == 1 ? greenFill : day_dir == -1 ? redFill : invis)


//Running outside of OR Session
if not or_sesh and or_token
    h_ln.ln.set_x2(bar_index)
    l_ln.ln.set_x2(bar_index)
    h_ln.lab.set_x(bar_index)
    l_ln.lab.set_x(bar_index)
    m_ln.set_x2(bar_index)

// MODIFIED: Check if first targets are hit
if not na(first_up_target) and h_src > first_up_target and not first_up_hit
    first_up_hit := true
    hst := h_src

if not na(first_down_target) and l_src < first_down_target and not first_down_hit
    first_down_hit := true
    lst := l_src

// MODIFIED: Update subsequent targets if needed
if first_up_hit and h_src > hst
    hst := h_src
if first_down_hit and l_src < lst
    lst := l_src

// Calculate maximum targets based on subsequent targets spacing
up_max = first_up_hit ? get_1up((hst - first_up_target)/(orw*tSubsequentPer)) : 0
down_max = first_down_hit ? get_1up((first_down_target - lst)/(orw*tSubsequentPer)) : 0

// Calculate current target level for subsequent targets
up_cur = first_up_hit ? math.max(0,get_1up((h_src - first_up_target)/(orw*tSubsequentPer))) : 0
down_cur = first_down_hit ? math.max(0,get_1up((first_down_target - l_src)/(orw*tSubsequentPer))) : 0

//Signal Calcs
if  (close > orm and down_check == false)
    down_check := true

xdown = ta.crossunder(close,orl)
xdown2 = ta.crossunder(close,orl-orw*tSubsequentPer)
if (useBias ? ((day_dir != 1 and xdown) or (day_dir == 1 and xdown2)) : ta.crossunder(close,orl)) and down_check
    down_signal := true
    down_check := false

if (close < orm and up_check == false)
    up_check := true

xup = ta.crossover(close,orh)
xup2 = ta.crossover(close,orh+orw*tSubsequentPer)
if (useBias ? ((day_dir != -1 and xup) or (day_dir == -1 and xup2)) : ta.crossover(close,orh)) and up_check
    up_signal := true
    up_check := false

//---------------------------------------------------------------------------------------------------------------------}
//Display
//---------------------------------------------------------------------------------------------------------------------{

//Targets
if not or_sesh and or_token
    // MODIFIED: Draw first target based on mid-range percentage
    if tTog and array.size(up_targs) == 0 and not na(first_up_target)
        up_targs.push(target.new(
          line.new(bar_index-1,first_up_target,bar_index,first_up_target, color = green, style = linestyle(tStyle)),
          label.new(bar_index,first_up_target,text = "T1",tooltip = str.tostring(first_up_target,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(green,0), size = txtSize)
          ))
    
    if tTog and array.size(down_targs) == 0 and not na(first_down_target)
        down_targs.push(target.new(
          line.new(bar_index-1,first_down_target,bar_index,first_down_target, color = red, style = linestyle(tStyle)),
          label.new(bar_index,first_down_target,text = "T1",tooltip = str.tostring(first_down_target,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(red,0), size = txtSize)
          ))
    
    // MODIFIED: Draw subsequent targets based on OR width percentage
    if first_up_hit and up_count < up_max and tTog
        for i = (up_count+1) to up_max 
            up_targs.push(target.new(
              line.new(bar_index-1,(first_up_target+orw*tSubsequentPer*i),bar_index,(first_up_target+(orw*tSubsequentPer*i)), color = green, style = linestyle(tStyle)),
              label.new(bar_index,(first_up_target+orw*tSubsequentPer*i),text = str.tostring(i+1),tooltip = str.tostring(first_up_target+orw*tSubsequentPer*i,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(green,0), size = txtSize)
              ))
            if i == up_max
                up_count := up_max
    if first_down_hit and down_count < down_max and tTog
        for i = (down_count+1) to down_max
            down_targs.push(target.new(
              line.new(bar_index-1,(first_down_target-orw*tSubsequentPer*i),bar_index,(first_down_target-(orw*tSubsequentPer*i)), color = red, style = linestyle(tStyle)),
              label.new(bar_index,(first_down_target-orw*tSubsequentPer*i),text = str.tostring(i+1),tooltip = str.tostring(first_down_target-orw*tSubsequentPer*i,format.mintick), style = label.style_label_left, color = invis, textcolor = color.new(red,0), size = txtSize)
              ))
            if i == down_max
                down_count := down_max

//Extending to Current Bar   
if tDispType == "Extended" and tTog
    for targ in up_targs
        targ.ln.set_x2(bar_index)
        targ.lab.set_x(bar_index)
    for targ in down_targs
        targ.ln.set_x2(bar_index)
        targ.lab.set_x(bar_index) 

if tDispType == "Adaptive" and tTog
    for targ in up_targs
        if targ.ln.get_y1() <= (first_up_target+(orw*tSubsequentPer*(up_cur))) and targ.ln.get_y1() >= (first_up_target+(orw*tSubsequentPer*(up_cur-2)))
            targ.ln.set_x2(bar_index+1)
            targ.lab.set_x(bar_index+1)
    for targ in down_targs
        if targ.ln.get_y1() <= (first_down_target-(orw*tSubsequentPer*(down_cur-2))) and targ.ln.get_y1() >= (first_down_target-(orw*tSubsequentPer*(down_cur)))
            targ.ln.set_x2(bar_index+1)
            targ.lab.set_x(bar_index+1)


//Moving Average
ma = day_ma(or_start,maType,close,maLen)
plot(or_start?na:ma, style = plot.style_linebr, color = maColor, display = maTog?display.all:display.none, title = "Moving Average", editable = false)

//Signals
if up_signal and sigTog
    signals.push(label.new(bar_index, orl,
      style = label.style_label_center, 
      text = "\n▲", 
      color = invis, 
      textcolor = upSigColor, 
      size = sigSize))

if down_signal and sigTog
    signals.push(label.new(bar_index, orh,
      style = label.style_label_center, 
      text = "▼\n",
      color = invis, 
      textcolor = downSigColor, 
      size = sigSize))

//---------------------------------------------------------------------------------------------------------------------}
if barstate.islast
    if not na(h_ln.lab)
        h_ln.lab.set_text("ORH  " + str.tostring(h_ln.ln.get_y1(), format.mintick))
        h_ln.lab.set_color(color.new(orColor, 0))
        h_ln.lab.set_textcolor(color.white)

    if not na(l_ln.lab)
        l_ln.lab.set_text("ORL  " + str.tostring(l_ln.ln.get_y1(), format.mintick))
        l_ln.lab.set_color(color.new(orColor, 0))
        l_ln.lab.set_textcolor(color.white)

    if tTog
        for [i, targ] in up_targs
            targ.lab.set_text("T" + str.tostring(i + 1) + "  " + str.tostring(targ.ln.get_y1(), format.mintick))
            targ.lab.set_color(color.new(green, 0))
            targ.lab.set_textcolor(color.white)

        for [i, targ] in down_targs
            targ.lab.set_text("T" + str.tostring(i + 1) + "  " + str.tostring(targ.ln.get_y1(), format.mintick))
            targ.lab.set_color(color.new(red, 0))
            targ.lab.set_textcolor(color.white)




            ##################################### ABOVE IS THE PINESCRIPT CODE##################################

    HERE ARE THE TRADES. | Instrument | Size | Price | Direction | Type | Fee | Bot ID | Label | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-PERPETUAL | 0,1065 | 64633 | buy | liquidation | 3,4439 |  |  | 2026-11-15 00:49:05 |
| BTC-PERPETUAL | 0,0001 | 64808 | buy | liquidation | 0,0032 |  |  | 2026-11-15 00:48:23 |
| BTC-PERPETUAL | 0,0001 | 64795 | buy | liquidation | 0,0032 |  |  | 2026-11-15 00:48:23 |
| BTC-PERPETUAL | 0,0001 | 64782 | buy | liquidation | 0,0032 |  |  | 2026-11-15 00:48:23 |
| BTC-PERPETUAL | 0,0001 | 64769 | buy | liquidation | 0,0032 |  |  | 2026-11-15 00:48:23 |
| BTC-PERPETUAL | 0,0001 | 64756 | buy | liquidation | 0,0032 |  |  | 2026-11-15 00:48:23 |
| BTC-27NOV26-75000-C | 0,01 | 35 | sell | liquidation | 0,0438 |  |  | 2026-11-15 00:48:13 |
| BTC-25DEC26-68000-C | 0,01 | 2135 | sell | liquidation | 0,097 |  |  | 2026-11-15 00:48:09 |
| BTC-25DEC26-80000-C | 0,2 | 195 | sell | liquidation | 1,9408 |  |  | 2026-11-15 00:47:59 |
| BTC-25DEC26-70000-C | 0,02 | 1460 | sell | liquidation | 0,1941 |  |  | 2026-11-15 00:47:55 |
| BTC-25DEC26-65000-C | 0,03 | 3535 | sell | liquidation | 0,2911 |  |  | 2026-11-15 00:47:50 |
| BTC-25DEC26-73000-C | 0,12 | 770 | sell | liquidation | 1,1651 |  |  | 2026-11-15 00:47:41 |
| BTC-27NOV26-87000-P | 0,1 | 22225 | buy | liquidation | 0,9711 |  |  | 2026-11-15 00:47:26 |
| BTC-PERPETUAL | 0,0148 | 65422 | buy | normal | 0,0968 |  |  | 2026-11-14 19:02:57 |
| BTC-PERPETUAL | 0,0076 | 65422 | buy | normal | 0,0497 |  |  | 2026-11-14 19:02:57 |
| BTC-PERPETUAL | 0,0076 | 65422 | buy | normal | 0,0497 |  |  | 2026-11-14 19:02:57 |
| BTC-PERPETUAL | 0,0001 | 64833 | buy | normal | 0,0032 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0001 | 64820 | buy | normal | 0,0032 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0001 | 64807 | buy | normal | 0,0032 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0001 | 64794 | buy | normal | 0,0032 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0001 | 64781 | buy | normal | 0,0032 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0009 | 64704 | buy | normal | 0,0291 |  |  | 2026-11-14 18:46:06 |
| BTC-PERPETUAL | 0,0001 | 64856 | buy | normal | 0,0032 |  |  | 2026-11-14 18:44:31 |
| BTC-PERPETUAL | 0,0001 | 64843 | buy | normal | 0,0032 |  |  | 2026-11-14 18:44:31 |
| BTC-PERPETUAL | 0,0001 | 64830 | buy | normal | 0,0032 |  |  | 2026-11-14 18:44:31 |
| BTC-PERPETUAL | 0,0001 | 64817 | buy | normal | 0,0032 |  |  | 2026-11-14 18:44:31 |
| BTC-PERPETUAL | 0,0001 | 64804 | buy | normal | 0,0032 |  |  | 2026-11-14 18:44:31 |
| BTC-PERPETUAL | 0,0009 | 64719 | buy | normal | 0,0291 |  |  | 2026-11-14 18:44:31 |
| BTC-25DEC26-65000-C | 0,02 | 3300 | buy | normal | 0,1912 |  |  | 2026-11-14 14:39:35 |
| BTC-PERPETUAL | 0,03 | 64335 | sell | normal | 0,9658 |  |  | 2026-11-14 13:25:55 |
| BTC-PERPETUAL | 0,05 | 64127 | sell | normal | 1,605 |  |  | 2026-11-14 13:16:43 |
| BTC-25DEC26-65000-C | 0,01 | 3730 | buy | normal | 0,0965 |  |  | 2026-11-14 13:15:23 |
| BTC-PERPETUAL | 0,01 | 64536 | buy | normal | 0,3229 |  |  | 2026-11-14 13:06:23 |
| BTC-PERPETUAL | 0,01 | 64601 | buy | normal | 0,323 |  |  | 2026-11-14 12:54:41 |
| BTC-PERPETUAL | 0,01 | 64484 | buy | normal | 0,3227 |  |  | 2026-11-14 12:54:36 |
| BTC-PERPETUAL | 0,01 | 64459 | buy | normal | 0,3227 |  |  | 2026-11-14 12:54:35 |
| BTC-PERPETUAL | 0,01 | 65343 | buy | normal | 0,3269 |  |  | 2026-11-14 07:20:47 |
| BTC-PERPETUAL | 0,01 | 65996 | buy | normal | 0,3302 |  |  | 2026-11-14 06:26:45 |
| BTC-25DEC26-80000-C | 0,1 | 370 | buy | normal | 0,9976 |  |  | 2026-11-14 06:17:52 |
| BTC-25DEC26-68000-C | 0,01 | 3035 | buy | normal | 0,0997 |  |  | 2026-11-14 06:11:48 |
| BTC-PERPETUAL | 0,01 | 66289 | buy | normal | 0,3316 |  |  | 2026-11-14 06:02:28 |
| BTC-PERPETUAL | 0,01 | 66500 | buy | normal | 0,3328 |  |  | 2026-11-13 20:14:12 |
| BTC-PERPETUAL | 0,03 | 66837 | buy | normal | 1,0034 |  |  | 2026-11-13 19:35:57 |
| BTC-PERPETUAL | 0,02 | 67993 | sell | normal | 0,6806 |  |  | 2026-11-13 13:44:28 |
| BTC-PERPETUAL | 0,02 | 68749 | buy | normal | 0,6881 |  |  | 2026-11-13 08:00:50 |
| BTC-PERPETUAL | 0,0191 | 68711 | sell | normal | 0,6569 |  |  | 2026-11-13 07:19:04 |
| BTC-PERPETUAL | 0,0009 | 68713 | sell | normal | 0,031 |  |  | 2026-11-13 07:19:04 |
| BTC-13NOV26-73000-C | 0,02 | 0 | sell | expiration | 0 |  |  | 2026-11-13 00:00:00 |
| BTC-13NOV26-90000-C | 0,02 | 0 | buy | expiration | 0 |  |  | 2026-11-13 00:00:00 |
| BTC-PERPETUAL | 0,0491 | 68702 | sell | normal | 1,6863 |  |  | 2026-11-12 20:53:14 |
| BTC-PERPETUAL | 0,0009 | 68703 | sell | normal | 0,0309 |  |  | 2026-11-12 20:53:14 |
| BTC-PERPETUAL | 0,02 | 68811 | buy | normal | 0,6876 |  |  | 2026-11-12 19:55:06 |
| BTC-25DEC26-70000-C | 0,02 | 3450 | buy | normal | 0,207 |  |  | 2026-11-12 19:41:29 |
| BTC-PERPETUAL | 0,02 | 69307 | buy | normal | 0,6926 |  |  | 2026-11-12 19:24:13 |
| BTC-25DEC26-73000-C | 0,01 | 2415 | buy | normal | 0,1044 |  |  | 2026-11-12 19:18:04 |
| BTC-25DEC26-80000-C | 0,1 | 805 | buy | normal | 1,0487 |  |  | 2026-11-12 19:13:01 |
| BTC-PERPETUAL | 0,0191 | 70225 | sell | normal | 0,6705 |  |  | 2026-11-12 18:57:30 |
| BTC-PERPETUAL | 0,0009 | 70231 | sell | normal | 0,0316 |  |  | 2026-11-12 18:57:30 |
| BTC-25DEC26-73000-C | 0,01 | 2670 | buy | normal | 0,1054 |  |  | 2026-11-12 18:55:26 |
| BTC-25DEC26-73000-C | 0,04 | 3045 | buy | normal | 0,4263 |  |  | 2026-11-12 16:56:04 |
| BTC-PERPETUAL | 0,0991 | 71038 | sell | normal | 3,5188 |  |  | 2026-11-12 16:55:18 |
| BTC-PERPETUAL | 0,0009 | 71040 | sell | normal | 0,032 |  |  | 2026-11-12 16:55:18 |
| BTC-PERPETUAL | 0,01 | 70802 | buy | normal | 0,3539 |  |  | 2026-11-12 16:38:57 |
| BTC-25DEC26-73000-C | 0,01 | 2975 | buy | normal | 0,1063 |  |  | 2026-11-12 16:37:21 |
| BTC-PERPETUAL | 0,01 | 71386 | buy | normal | 0,3567 |  |  | 2026-11-12 06:55:06 |
| BTC-PERPETUAL | 0,01 | 71379 | buy | normal | 0,3566 |  |  | 2026-11-12 06:55:05 |
| BTC-25DEC26-73000-C | 0,01 | 3330 | buy | normal | 0,1073 |  |  | 2026-11-12 06:50:14 |
| BTC-PERPETUAL | 0,0089 | 71969 | buy | normal | 0,3199 |  |  | 2026-11-12 06:21:50 |
| BTC-PERPETUAL | 0,0001 | 71965 | buy | normal | 0,0036 |  |  | 2026-11-12 06:21:50 |
| BTC-PERPETUAL | 0,02 | 71954 | buy | normal | 0,719 |  |  | 2026-11-12 06:21:50 |
| BTC-PERPETUAL | 0,0001 | 71950 | buy | normal | 0,0036 |  |  | 2026-11-12 06:21:50 |
| BTC-PERPETUAL | 0,0009 | 71915 | buy | normal | 0,0324 |  |  | 2026-11-12 06:21:50 |
| BTC-PERPETUAL | 0,01 | 72191 | buy | normal | 0,3608 |  |  | 2026-11-12 06:11:11 |
| BTC-PERPETUAL | 0,02 | 72531 | buy | normal | 0,725 |  |  | 2026-11-12 05:45:02 |
| BTC-PERPETUAL | 0,02 | 73606 | buy | normal | 0,7358 |  |  | 2026-11-11 17:26:31 |
| BTC-PERPETUAL | 0,0199 | 74003 | sell | normal | 0,736 |  |  | 2026-11-11 14:43:19 |
| BTC-PERPETUAL | 0,0291 | 74064 | sell | normal | 1,077 |  |  | 2026-11-11 14:37:14 |
| BTC-PERPETUAL | 0,0009 | 74066 | sell | normal | 0,0333 |  |  | 2026-11-11 14:37:14 |
| BTC-PERPETUAL | 0,0001 | 74101 | buy | normal | 0,0037 |  |  | 2026-11-11 14:37:00 |
| BTC-PERPETUAL | 0,0191 | 74378 | sell | normal | 0,7102 |  |  | 2026-11-11 14:03:59 |
| BTC-PERPETUAL | 0,0009 | 74381 | sell | normal | 0,0335 |  |  | 2026-11-11 14:03:59 |
| BTC-PERPETUAL | 0,01 | 74663 | buy | normal | 0,3732 |  |  | 2026-11-11 09:19:21 |
| BTC-PERPETUAL | 0,01 | 73979 | buy | normal | 0,3698 |  |  | 2026-11-11 07:58:11 |
| BTC-PERPETUAL | 0,01 | 72751 | buy | normal | 0,3637 |  |  | 2026-11-10 08:37:56 |
| BTC-PERPETUAL | 0,0191 | 72873 | buy | normal | 0,6957 |  |  | 2026-11-10 08:36:35 |
| BTC-PERPETUAL | 0,0009 | 72868 | buy | normal | 0,0328 |  |  | 2026-11-10 08:36:35 |
| BTC-PERPETUAL | 0,1 | 73148 | sell | normal | 3,6568 |  |  | 2026-11-10 07:47:41 |
| BTC-PERPETUAL | 0,02 | 72911 | buy | normal | 0,7289 |  |  | 2026-11-10 07:44:35 |
| BTC-25DEC26-73000-C | 0,01 | 4230 | buy | normal | 0,1095 |  |  | 2026-11-10 07:42:13 |
| BTC-PERPETUAL | 0,02 | 73276 | buy | normal | 0,7323 |  |  | 2026-11-10 07:40:55 |
| BTC-25DEC26-73000-C | 0,02 | 4525 | buy | normal | 0,2207 |  |  | 2026-11-10 06:32:04 |
| BTC-PERPETUAL | 0,0491 | 73722 | sell | normal | 1,8093 |  |  | 2026-11-10 06:04:29 |
| BTC-PERPETUAL | 0,0009 | 73723 | sell | normal | 0,0332 |  |  | 2026-11-10 06:04:29 |
| BTC-PERPETUAL | 0,1191 | 73926 | buy | normal | 4,4008 |  |  | 2026-11-10 05:51:15 |
| BTC-PERPETUAL | 0,0009 | 73922 | buy | normal | 0,0333 |  |  | 2026-11-10 05:51:15 |
| BTC-PERPETUAL | 0,0191 | 74774 | buy | normal | 0,7138 |  |  | 2026-11-09 20:30:35 |
| BTC-PERPETUAL | 0,0009 | 74772 | buy | normal | 0,0336 |  |  | 2026-11-09 20:30:35 |
| BTC-PERPETUAL | 0,05 | 74962 | sell | normal | 1,8733 |  |  | 2026-11-09 18:02:22 |
| BTC-PERPETUAL | 0,0241 | 74936 | sell | normal | 0,9027 |  |  | 2026-11-08 16:46:03 |
| BTC-PERPETUAL | 0,0009 | 74939 | sell | normal | 0,0337 |  |  | 2026-11-08 16:46:03 |
| BTC-PERPETUAL | 0,05 | 75539 | sell | normal | 1,8878 |  |  | 2026-11-08 15:21:57 |
| BTC-27NOV26-70000-P | 0,1 | 1070 | sell | normal | 1,1127 |  |  | 2026-11-08 14:35:38 |
| BTC-25DEC26-73000-C | 0,01 | 5185 | buy | normal | 0,1113 |  |  | 2026-11-08 14:34:41 |
| BTC-PERPETUAL | 0,0141 | 74252 | buy | normal | 0,5233 |  |  | 2026-11-08 14:11:11 |
| BTC-PERPETUAL | 0,0009 | 74249 | buy | normal | 0,0334 |  |  | 2026-11-08 14:11:11 |
| BTC-27NOV26-75000-C | 0,01 | 2425 | buy | normal | 0,1115 |  |  | 2026-11-08 14:10:23 |
| BTC-PERPETUAL | 0,02 | 74962 | buy | normal | 0,7495 |  |  | 2026-11-06 07:05:02 |
| BTC-06NOV26-80000-P | 0,02 | 4522,96631924 | buy | expiration | 0,151 |  |  | 2026-11-06 00:00:00 |
| BTC-06NOV26-80000-C | 0,02 | 0 | buy | expiration | 0 |  |  | 2026-11-06 00:00:00 |
| BTC-PERPETUAL | 0,05 | 76025 | buy | normal | 1,8996 |  |  | 2026-11-05 07:59:21 |
| BTC-PERPETUAL | 0,1 | 79277 | sell | normal | 3,9627 |  |  | 2026-11-05 06:21:05 |
| BTC-30OCT26-81500-P | 0,05 | 3388,95403216 | sell | expiration | 0,3906 |  |  | 2026-10-30 01:00:00 |
| BTC-30OCT26-80500-P | 0,05 | 2388,95403216 | buy | expiration | 0,3906 |  |  | 2026-10-30 01:00:00 |
| BTC-30OCT26-80500-C | 0,05 | 0 | buy | expiration | 0 |  |  | 2026-10-30 01:00:00 |
| BTC-30OCT26-79500-C | 0,05 | 0 | sell | expiration | 0 |  |  | 2026-10-30 01:00:00 |
| BTC-13NOV26-73000-C | 0,02 | 5400 | buy | normal | 0,2303 |  |  | 2026-10-29 17:32:06 |
| BTC-13NOV26-90000-C | 0,02 | 140 | sell | normal | 0 |  |  | 2026-10-29 17:32:06 |
| BTC-27NOV26-70000-P | 0,1 | 755 | buy | normal | 1,1867 |  |  | 2026-10-29 14:15:03 |
| BTC-27NOV26-87000-P | 0,1 | 8180 | sell | normal | 0 |  |  | 2026-10-29 14:15:03 |
| BTC-06NOV26-80000-P | 0,02 | 1205 | sell | normal | 0,2428 |  |  | 2026-10-29 10:40:06 |
| BTC-06NOV26-80000-C | 0,02 | 2215 | sell | normal | 0 |  |  | 2026-10-29 10:40:06 |
| BTC-30OCT26-80500-C | 0,05 | 390 | sell | normal | 0,6006 |  |  | 2026-10-28 18:57:26 |
| BTC-30OCT26-79500-C | 0,05 | 1045 | buy | normal | 0 |  |  | 2026-10-28 18:57:26 |
| BTC-30OCT26-80500-P | 0,05 | 760 | sell | normal | 0 |  |  | 2026-10-28 18:57:26 |
| BTC-30OCT26-81500-P | 0,05 | 1670 | buy | normal | 0 |  |  | 2026-10-28 18:57:26 |
| BTC-30OCT26-83000-P | 0,01 | 3025 | buy | normal | 0,1203 |  |  | 2026-10-28 18:51:11 |
| BTC-30OCT26-81000-P | 0,01 | 985 | sell | normal | 0,1203 |  |  | 2026-10-28 18:51:08 |
| BTC-27NOV26-83000-P | 0,03 | 4360 | sell | normal | 0,3625 |  |  | 2026-10-28 10:25:56 |
| BTC-27NOV26-85000-P | 0,03 | 5830 | buy | normal | 0,3624 |  |  | 2026-10-28 10:25:51 |
| BTC-27NOV26-85000-P | 0,03 | 5660 | sell | normal | 0,3624 |  |  | 2026-10-28 09:57:56 |
| BTC-27NOV26-83000-P | 0,03 | 4485 | buy | normal | 0 |  |  | 2026-10-28 09:57:56 |
| BTC-30OCT26-83000-P | 0,01 | 2845 | sell | normal | 0,1204 |  |  | 2026-10-26 08:58:10 |
| BTC-30OCT26-81000-P | 0,01 | 1535 | buy | normal | 0 |  |  | 2026-10-26 08:58:10 |
| BTC-06NOV26-78000-C | 0,02 | 4085 | buy | normal | 0,2421 |  |  | 2026-10-24 17:02:00 |
| BTC-06NOV26-83000-C | 0,02 | 1250 | sell | normal | 0,2421 |  |  | 2026-10-24 17:01:55 |
| BTC-PERPETUAL | 0,0116 | 80735 | sell | normal | 0,4681 |  |  | 2026-10-24 17:01:51 |
| BTC-PERPETUAL | 0,0009 | 80738 | sell | normal | 0,0363 |  |  | 2026-10-24 17:01:51 |
| BTC-PERPETUAL | 0,017 | 80770 | sell | normal | 0,1373 |  |  | 2026-10-24 17:00:33 |
| BTC-PERPETUAL | 0,001 | 79378 | buy | normal | 0,0079 |  |  | 2026-10-24 08:27:14 |
| BTC-13NOV26-82000-C | 0,02 | 1790 | buy | normal | 0,2378 |  |  | 2026-10-24 08:20:26 |
| BTC-PERPETUAL | 0,01 | 79454 | buy | normal | 0,0794 |  |  | 2026-10-24 08:17:23 |
| BTC-PERPETUAL | 0,001 | 79831 | buy | normal | 0,008 |  |  | 2026-10-24 08:07:59 |
| BTC-PERPETUAL | 0,001 | 80078 | buy | normal | 0,008 |  |  | 2026-10-24 07:24:26 |
| BTC-PERPETUAL | 0,0165 | 81548 | buy | normal | 0,1345 | 0055DB5E04000002 |  | 2026-10-23 11:17:33 |
| BTC-13NOV26-82000-P | 0,02 | 2940 | buy | normal | 0,2443 |  |  | 2026-10-23 11:12:33 |
| BTC-13NOV26-82000-P | 0,02 | 2815 | sell | normal | 0,2445 |  |  | 2026-10-23 10:47:41 |
| BTC-13NOV26-82000-C | 0,02 | 2595 | sell | normal | 0 |  |  | 2026-10-23 10:47:41 |
| BTC-06NOV26-83000-C | 0,02 | 2490 | buy | normal | 0,2483 |  |  | 2026-10-21 20:09:01 |
| BTC-06NOV26-78000-C | 0,02 | 5675 | sell | normal | 0 |  |  | 2026-10-21 20:09:01 |
| BTC-23OCT26-88000-P | 0,03 | 3570 | buy | normal | 0,3809 |  |  | 2026-10-21 08:15:09 |
| BTC-23OCT26-85000-P | 0,03 | 1720 | sell | normal | 0,3771 |  |  | 2026-10-21 07:08:52 |
| BTC-23OCT26-85000-P | 0,03 | 3875 | buy | normal | 0,3666 |  |  | 2026-10-18 15:31:48 |
| BTC-06NOV26-83000-P | 0,03 | 3275 | sell | normal | 0,367 |  |  | 2026-10-18 15:14:15 |
| BTC-PERPETUAL | 0,0128 | 81565 | buy | normal | 0,5219 |  |  | 2026-10-18 15:14:02 |
| BTC-23OCT26-80000-P | 0,02 | 650 | sell | normal | 0,2446 |  |  | 2026-10-18 15:13:51 |
| BTC-23OCT26-87000-P | 0,01 | 6075 | buy | normal | 0,1218 |  |  | 2026-10-18 15:00:23 |
| BTC-PERPETUAL | 0,0293 | 80961 | buy | normal | 0,2371 | 0055CE1A04000001 |  | 2026-10-18 14:02:51 |
| BTC-06NOV26-85000-P | 0,03 | 5185 | buy | normal | 0,3639 |  |  | 2026-10-18 13:57:50 |
| BTC-23OCT26-80000-P | 0,02 | 970 | buy | normal | 0,2426 |  |  | 2026-10-18 13:55:45 |
| BTC-PERPETUAL | 0,0367 | 81830 | sell | normal | 0,3002 | 0055CE1A04000001 |  | 2026-10-17 21:02:08 |
| BTC-PERPETUAL | 0,0009 | 81838 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:07 |
| BTC-PERPETUAL | 0,0009 | 81823 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:06 |
| BTC-PERPETUAL | 0,0009 | 81822 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:05 |
| BTC-PERPETUAL | 0,0009 | 81823 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:04 |
| BTC-PERPETUAL | 0,0009 | 81822 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:03 |
| BTC-PERPETUAL | 0,0009 | 81824 | sell | normal | 0,0074 | 0055CE1A04000001 |  | 2026-10-17 21:02:02 |
| BTC-30OCT26-90000-C | 0,01 | 165 | buy | normal | 0,1228 |  |  | 2026-10-17 21:00:43 |
| BTC-30OCT26-87000-P | 0,01 | 5160 | sell | normal | 0,1228 |  |  | 2026-10-17 21:00:36 |
| BTC-30OCT26-84000-P | 0,02 | 3185 | buy | normal | 0,2455 |  |  | 2026-10-17 21:00:30 |
| BTC-30OCT26-80000-P | 0,02 | 1115 | sell | normal | 0,2455 |  |  | 2026-10-17 21:00:24 |
| BTC-30OCT26-80000-C | 0,01 | 3145 | sell | normal | 0,1228 |  |  | 2026-10-17 21:00:17 |
| BTC-23OCT26-87000-C | 0,01 | 45 | sell | normal | 0,0563 |  |  | 2026-10-17 19:51:26 |
| BTC-23OCT26-78000-P | 0,01 | 240 | sell | normal | 0,1225 |  |  | 2026-10-17 19:51:21 |
| BTC-23OCT26-83000-P | 0,03 | 1855 | sell | normal | 0,3673 |  |  | 2026-10-17 19:50:22 |
| BTC-23OCT26-82000-C | 0,01 | 1065 | buy | normal | 0,1224 |  |  | 2026-10-17 19:50:16 |
| BTC-30OCT26-90000-P | 0,01 | 8555 | buy | normal | 0,1224 |  |  | 2026-10-17 19:50:06 |
| BTC-06NOV26-85000-P | 0,01 | 4880 | sell | normal | 0,1215 |  |  | 2026-10-17 14:51:15 |
| BTC-06NOV26-83000-P | 0,01 | 3775 | buy | normal | 0 |  |  | 2026-10-17 14:51:15 |
| BTC-06NOV26-83000-P | 0,02 | 3780 | buy | normal | 0,2431 |  |  | 2026-10-17 14:45:01 |
| BTC-06NOV26-85000-P | 0,02 | 4880 | sell | normal | 0 |  |  | 2026-10-17 14:45:01 |
| BTC-27NOV26-110000-C | 0,01 | 85 | buy | normal | 0,1063 |  |  | 2026-10-16 02:04:37 |
| BTC-27NOV26-85000-P | 0,01 | 5185 | sell | normal | 0,1233 |  |  | 2026-10-15 22:15:52 |
| BTC-27NOV26-90000-P | 0,01 | 8935 | buy | normal | 0,1233 |  |  | 2026-10-15 22:15:34 |
| BTC-27NOV26-100000-C | 0,01 | 295 | sell | normal | 0,1233 |  |  | 2026-10-15 22:14:10 |
| BTC-30OCT26-80000-P | 0,02 | 2130 | buy | normal | 0,2405 |  |  | 2026-10-15 06:58:39 |
| BTC-30OCT26-84000-P | 0,02 | 4420 | sell | normal | 0 |  |  | 2026-10-15 06:58:39 |
| BTC-30OCT26-80000-C | 0,01 | 3045 | buy | normal | 0,1215 |  |  | 2026-10-14 20:20:33 |
| BTC-30OCT26-90000-C | 0,01 | 215 | sell | normal | 0 |  |  | 2026-10-14 20:20:33 |
| BTC-23OCT26-88000-P | 0,03 | 6770 | sell | normal | 0,3651 |  |  | 2026-10-14 19:50:29 |
| BTC-23OCT26-83000-P | 0,03 | 2805 | buy | normal | 0 |  |  | 2026-10-14 19:50:29 |
| BTC-23OCT26-87000-P | 0,01 | 5425 | sell | normal | 0,1223 |  |  | 2026-10-14 15:49:40 |
| BTC-23OCT26-83000-P | 0,01 | 2495 | buy | normal | 0 |  |  | 2026-10-14 15:49:40 |
| BTC-23OCT26-87000-C | 0,01 | 275 | buy | normal | 0,1224 |  |  | 2026-10-14 09:04:53 |
| BTC-23OCT26-78000-P | 0,01 | 580 | buy | normal | 0 |  |  | 2026-10-14 09:04:53 |
| BTC-23OCT26-83000-P | 0,01 | 2300 | sell | normal | 0 |  |  | 2026-10-14 09:04:53 |
| BTC-23OCT26-82000-C | 0,01 | 1455 | sell | normal | 0 |  |  | 2026-10-14 09:04:53 |
| BTC-27NOV26-100000-C | 0,01 | 460 | buy | normal | 0,1229 |  |  | 2026-10-14 07:13:50 |
| BTC-27NOV26-110000-C | 0,01 | 105 | sell | normal | 0 |  |  | 2026-10-14 07:13:50 |
| BTC-30OCT26-90000-P | 0,01 | 7025 | sell | normal | 0,1244 |  |  | 2026-10-12 19:57:45 |
| BTC-30OCT26-87000-P | 0,01 | 4870 | buy | normal | 0 |  |  | 2026-10-12 19:57:45 |
| BTC-27NOV26-90000-P | 0,01 | 8300 | sell | normal | 0,1241 |  |  | 2026-10-12 19:50:25 |
| BTC-27NOV26-85000-P | 0,01 | 5250 | buy | normal | 0 |  |  | 2026-10-12 19:50:25 |
| BTC-30OCT26-73000-P | 0,01 | 330 | buy | normal | 0,1242 |  |  | 2026-10-12 06:58:46 |
| BTC-30OCT26-65000-P | 0,01 | 75 | buy | normal | 0,0938 |  |  | 2026-10-12 06:58:14 |
| BTC-30OCT26-78000-P | 0,02 | 785 | sell | normal | 0,2484 |  |  | 2026-10-12 06:55:07 |
| BTC-23OCT26-75000-P | 0,02 | 230 | buy | normal | 0,2478 |  |  | 2026-10-12 06:41:46 |
| BTC-23OCT26-90000-P | 0,02 | 7210 | sell | normal | 0,248 |  |  | 2026-10-12 06:40:22 |
| BTC-23OCT26-90000-C | 0,02 | 155 | sell | normal | 0,248 |  |  | 2026-10-12 06:40:11 |
| BTC-23OCT26-75000-C | 0,02 | 8240 | buy | normal | 0,248 |  |  | 2026-10-12 06:40:07 |
| BTC-23OCT26-90000-C | 0,02 | 345 | buy | normal | 0,2519 |  |  | 2026-10-11 22:28:05 |
| BTC-23OCT26-75000-C | 0,02 | 9115 | sell | normal | 0 |  |  | 2026-10-11 22:28:05 |
| BTC-23OCT26-90000-P | 0,02 | 6365 | buy | normal | 0,2521 |  |  | 2026-10-11 22:23:08 |
| BTC-23OCT26-75000-P | 0,02 | 175 | sell | normal | 0 |  |  | 2026-10-11 22:23:08 |
| BTC-30OCT26-78000-P | 0,01 | 915 | buy | normal | 0,1246 |  |  | 2026-10-09 07:27:10 |
| BTC-30OCT26-65000-P | 0,01 | 65 | sell | normal | 0 |  |  | 2026-10-09 07:27:10 |
| BTC-30OCT26-78000-P | 0,01 | 915 | buy | normal | 0,1246 |  |  | 2026-10-09 07:22:28 |
| BTC-30OCT26-73000-P | 0,01 | 310 | sell | normal | 0 |  |  | 2026-10-09 07:22:28 |
| BTC-16OCT26-80000-C | 0,04 | 2240 | buy | normal | 0,4849 |  |  | 2026-10-08 10:42:44 |
| BTC-16OCT26-75000-P | 0,05 | 295 | buy | normal | 0,6059 |  |  | 2026-10-08 10:42:33 |
| BTC-16OCT26-86000-C | 0,01 | 265 | sell | normal | 0,1212 |  |  | 2026-10-08 10:40:43 |
| BTC-16OCT26-83000-C | 0,04 | 800 | sell | normal | 0,485 |  |  | 2026-10-08 10:40:40 |
| BTC-16OCT26-82000-P | 0,04 | 2175 | sell | normal | 0,4849 |  |  | 2026-10-08 10:40:33 |
| BTC-16OCT26-83000-P | 0,01 | 2815 | sell | normal | 0,1212 |  |  | 2026-10-08 10:40:30 |
| BTC-16OCT26-84000-C | 0,01 | 645 | buy | normal | 0,1213 |  |  | 2026-10-08 10:40:26 |
| BTC-16OCT26-87000-C | 0,01 | 495 | sell | normal | 0,1252 |  |  | 2026-10-07 09:11:41 |
| BTC-30OCT26-85000-C | 0,01 | 2395 | buy | normal | 0,1254 |  |  | 2026-10-07 09:06:59 |
| BTC-16OCT26-83000-C | 0,04 | 2450 | buy | normal | 0,5044 |  |  | 2026-10-06 20:18:51 |
| BTC-16OCT26-80000-C | 0,04 | 4415 | sell | normal | 0 |  |  | 2026-10-06 20:18:51 |
| BTC-16OCT26-82000-P | 0,04 | 925 | buy | normal | 0,503 |  |  | 2026-10-06 19:53:06 |
| BTC-16OCT26-75000-P | 0,04 | 65 | sell | normal | 0 |  |  | 2026-10-06 19:53:06 |
| BTC-16OCT26-86000-C | 0,01 | 2150 | buy | normal | 0,1296 |  |  | 2026-10-06 07:54:34 |
| BTC-16OCT26-84000-C | 0,01 | 3275 | sell | normal | 0 |  |  | 2026-10-06 07:54:34 |
| BTC-16OCT26-83000-P | 0,01 | 980 | buy | normal | 0,1279 |  |  | 2026-10-04 08:35:39 |
| BTC-16OCT26-75000-P | 0,01 | 65 | sell | normal | 0 |  |  | 2026-10-04 08:35:39 |
| BTC-30OCT26-85000-P | 0,01 | 2295 | buy | normal | 0,1304 |  |  | 2026-10-02 06:47:50 |
| BTC-02OCT26-73000-P | 0,01 | 0 | buy | expiration | 0 |  |  | 2026-10-02 01:00:00 |
| BTC-02OCT26-75000-P | 0,01 | 0 | buy | expiration | 0 |  |  | 2026-10-02 01:00:00 |
| BTC-16OCT26-87000-C | 0,01 | 1135 | buy | normal | 0,1254 |  |  | 2026-09-30 07:38:42 |
| BTC-30OCT26-85000-P | 0,01 | 3395 | sell | normal | 0,1265 |  |  | 2026-09-30 07:07:57 |
| BTC-30OCT26-85000-C | 0,01 | 3120 | sell | normal | 0 |  |  | 2026-09-30 07:07:57 |
| BTC-30OCT26-110000-C | 0,02 | 95 | buy | normal | 0,2375 |  |  | 2026-09-29 06:37:41 |
| BTC-02OCT26-81000-P | 0,01 | 145 | buy | normal | 0,1261 |  |  | 2026-09-29 06:34:51 |
| BTC-02OCT26-83000-P | 0,02 | 420 | sell | normal | 0,2528 |  |  | 2026-09-29 05:54:14 |
| BTC-30OCT26-100000-C | 0,02 | 275 | sell | normal | 0,2528 |  |  | 2026-09-29 05:54:11 |
| BTC-02OCT26-82000-P | 0,02 | 205 | sell | normal | 0,2528 |  |  | 2026-09-29 05:54:08 |
| BTC-02OCT26-80000-P | 0,01 | 80 | buy | normal | 0,1 |  |  | 2026-09-29 05:53:36 |
| BTC-02OCT26-82000-P | 0,01 | 710 | buy | normal | 0,1245 |  |  | 2026-09-28 18:43:09 |
| BTC-02OCT26-75000-P | 0,01 | 20 | sell | normal | 0 |  |  | 2026-09-28 18:43:09 |
| BTC-02OCT26-75000-P | 0,01 | 35 | buy | normal | 0,0438 |  |  | 2026-09-28 15:58:08 |
| BTC-02OCT26-80000-C | 0,01 | 3845 | buy | normal | 0,1252 |  |  | 2026-09-28 15:55:40 |