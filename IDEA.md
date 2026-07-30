this is a project about a novel approach to finding seasonal components in times series. the goal is to provides an easy way to find multiple seasonal periods and patterns in a time series.

here are the main capabilities expected from the tool (others "add-on" capabilities will be explored):
- return one or multiple seasonal periods from a times
- extract the effect of each of these seasonal periods

that's it! adds-on will be:
- extracting trend, residuals
- detecting which type of seasonality is involved
- visually representing the seasonalities
- expend to non-integer seasonal period
- expend to calendar seasonal components

so here is the technique i am using: it all based on brute force. let's say we have a time series of length n. the idea is simply to test every periods from 1 to n (in fact n//2 when we are only interested in integer seasonal periods) to see if they are a seasonal component. the problem is to find a test that allow to conclude if a given number is a seasonal component. does such a test exist? (may need to do research on that). however i an came up with a way to do that (no proof, but empirically works so far that's why i find it interesting): anova.
the setup is the following: we assume an already detrended series of length n, so we only have seasonality and error (for the sake of the example, we assume error is null). we take a candidate seasons s and we "fold" the series into a matrix of shape of x lines and s columns and we run an anova on the columns and conclude from the p-value.
for example, we have the series l = 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 and we want to test if s=3 is a seasonality. then we fold l into 
lfold = 1, 2, 3
/       1, 2, 3
/       1, 2, 3
/       1, 2, 3
(4x3)
and then we run anova considering the groups [1,1,1,1], [2,2,2,2], [3,3,3,3], that will lead us to reject the null hypothesis (low p-values). and thus we conclude that s=3 is a seasonal period.


for the future
ofc, apart from t he limitation that there i know no proofs (nor can i provide one right now) that show the effectiveness of this methods (except from conclusive empirical results), we also have the fact that we need a detrended series, which poses other questions for the future of the project:
- how do we define a detrended series
- howcan we extend our method to series that are not detrended? OR (and maybe better?) how can we detrend any series to make sure that we can use our method as is?
- either way, how do we make sure that our method can handle both multiplicative and additive seasonality expecially if we have to detrend (make sure we do not remove seasonality but remove all trend)
