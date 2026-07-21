# Task 2 — Predicting Electricity Demand with an MLP

**Problem.** A power company wants to forecast daily electricity demand for the next 5 days,
using 5 years of past daily demand (values roughly between 80 and 400).

---

## a. Using an MLP for the prediction

I'd basically reuse the trick from Task 1. It's a time series, so instead of trying to
predict "the future" all at once, I'd chop the history into a sliding window: show the network
the last few days of demand and have it guess what comes next, then slide the window forward a
day and do it again. That turns five years of data into a big pile of training examples pretty
much for free.

The parts worth actually thinking about are how far back to look and how far ahead to predict.
Demand has an obvious weekly pattern — weekdays and weekends just aren't the same — so the
input window needs to cover at least a week. Somewhere around 7 to 14 days feels reasonable;
much shorter and the network literally can't see the weekly cycle. It'd probably also help to
throw in the day of the week or the month as extra inputs, so it doesn't have to figure the
calendar out from the raw numbers on its own.

For the output I don't think there's one right answer. The easy option is five output neurons,
one per day, and predict all five at once. The other option is to predict just tomorrow and
then feed that guess back in to step forward five times — which is really the recurrent idea
from Task 4. Predicting all five at once keeps the errors from piling up on each other, while
the step-by-step version is more flexible but tends to drift the further out it goes. I'd go
with the five-outputs version first since it's easier to get my head around.

The rest is the standard setup: one hidden layer to start (maybe 5–20 neurons, tuned on a
validation set), a **linear** output because we're predicting a number and not a yes/no,
sum-of-squares error, and a smallish learning rate with some momentum. Two things I wouldn't
skip: normalising the demand values (leaving them at 80–400 would let them dominate the
weights), and keeping a validation set with early stopping so it learns the pattern instead of
just memorising the training years.

## b. Adding the weather forecast

This one's easy. Temperature clearly affects demand — people crank the heating or the AC — so
the forecast temperatures are exactly the kind of hint the network wants. Since the MLP is
just a function of whatever inputs you give it, I'd tack the daytime and nighttime
temperatures on as two more input neurons next to the demand window, normalise them the same
way, and let training work out how much they count. Nothing about the network really changes;
the input layer just gets a bit wider. If we're predicting all five days, we'd add the
temperatures for each of those days the same way.

## c. Would it actually work?

For normal days, I think it'd do alright — the weekly cycle, the seasonal shift, and the
temperature effect are all steady, repeating patterns, and with five years of data it gets to
see loads of them. Tomorrow's prediction should be the most trustworthy.

But the catch is that it can only really predict stuff that looks like what it already saw
while training, and that's where it starts to struggle. One-off events are the big one:
holidays, a huge sporting final, a random early cold snap — there's no way for it to see those
coming unless we spell it out with something like a "holiday" flag. It's also bad at
extrapolating, so anything outside the range it trained on — a record heatwave, or demand
slowly creeping up over the years — it'll handle poorly. And the forecast just gets shakier the
further ahead you go: day five is leaning on a weather forecast that's already uncertain, and
if you're doing the step-by-step version the mistakes stack up along the way.

So it's a decent fit for regular, weather-driven demand, but I wouldn't trust it on the rare or
extreme days — which are usually the exact days a power company cares about most.
