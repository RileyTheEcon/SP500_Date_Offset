---
license: cc-by-4.0
language:
- en
tags:
- finance
- economics
- time series
pretty_name: 'S&P 500 Date Offset'
---
# S&P 500 Date Offset

Financial markets and the economy go hand-in-hand: we expect positive growth during good economic times and corrections during recessions. That said, at any given time, the investor or trader cannot have perfect information about the current state of the economy. Since there is often a several-months lag between when economic conditions are experienced and when they are officially reported, an amount of speculation gets priced into the market. The accuracy of this speculation is checked when the official economic data is finally released, often resulting in price swings when expectations and reality are not in line.
While this begs the important question as to whether this over- or under-confidence within the market can be measured, issues with data formatting persist. It is standard practice for economic data to be back-dated to reflect the first date of the time period it is meant to reflect, rather than the day of publication (ie CPI data for January is released in mid-February and GDP data for Q1 is released in late-April, but both are dated to January 1st).
This creates a significant problem: it is not enough to pull together data from various APIs and merge along a date index. This project seeks to off-set the economic data to reflect those dates 

This dataset could be the first step in modeling and eventually forecasting market expectations of economic publications and its reaction post-release.

