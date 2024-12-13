import numpy as np
import pandas as pd

#bold print statements
class color:
  BOLD = '\033[1m'
  END = '\033[0m'
#chisquare function
def calculate_chisquare(expected, observed, buckettype = 'bins', buckets = 10, axis = 0):
  '''Calculate the Chi-Square across all variables

    Args:
       expected: numpy matrix of original values
       observed: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       chi_square_values: ndarray of chi-square values for each variable

    '''

  def chi_square (expected_array, observed_array, buckets):
    '''Calculate the Chi-Square for a single variable

        Args:
           expected_array: numpy array of original values
           observed_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           chi_square_value: calculated Chi-Square value
        '''
    # Create the buckets, by establishing the scale range and breakpoints.
    def scale_range (input, min, max):
      input += -(np.min(input))
      input /= np.max(input) / (max - min)
      input += min
      return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
      breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
      breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    observed_percents = np.histogram(observed_array, breakpoints)[0] / len(observed_array)

    def sub_chisquare(e_perc, o_perc):
      '''Calculate the observed Chi-Square value from comparing the values.
         Update the observed value to a very small number if equal to zero.
      '''
      if o_perc == 0:
        o_perc = 0.0001
      if e_perc == 0:
        e_perc = 0.0001

      value = ((e_perc - o_perc)**2)/e_perc
      return(value)

    # Calculate the overall chi-square value.
    chi_square_value = np.sum(sub_chisquare(expected_percents[i], observed_percents[i]) for i in range(0, len(expected_percents)))
    return(chi_square_value)

  if len(expected.shape) == 1:
    chi_square_values = np.empty(len(expected.shape))
  else:
    chi_square_values = np.empty(expected.shape[axis])

  # Adapt the calculation to the axis: vertical or horizontal.
  for i in range(0, len(chi_square_values)):
    if len(chi_square_values) == 1:
      chi_square_values = chi_square(expected, observed, buckets)
    elif axis == 0:
      chi_square_values[i] = chi_square(expected[:,i], observed[:,i], buckets)
    elif axis == 1:
      chi_square_values[i] = chi_square(expected[i,:], observed[i,:], buckets)
  #print("Observed Chi-Square =", chi_square_values)
  return(chi_square_values)

def calculate_chi(expected, observed, buckettype = 'bins', buckets = 10, axis = 0):
  '''Runs a complete chi-square study, including chi-square value and p-value from the data.

    Args:
       expected: numpy matrix of original values
       observed: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       p_value: calculated p-value
       observed_calculated_chisquare: chi-square from the data
       critical_value: critical values from the distribution


    '''
  # Create bootstrap sample
  data = pd.DataFrame([])
  for i in np.arange(1000):
    bs_sample = np.random.choice(expected,len(expected))
    calculated_chi=calculate_chisquare(expected, bs_sample, buckettype, buckets, axis)
    data = data.append(pd.DataFrame({'calculated_chi_square': calculated_chi}, index=[0]), ignore_index=True)

  # Get critical values
  ## 5%
  critical_value_05_initial = data.quantile(0.95)
  critical_value_05_final = critical_value_05_initial['calculated_chi_square']

  ## 1%
  critical_value_01_initial = data.quantile(0.99)
  critical_value_01_final = critical_value_01_initial['calculated_chi_square']

  ## 0.01%
  critical_value_001_initial = data.quantile(0.999)
  critical_value_001_final = critical_value_001_initial['calculated_chi_square']

  # Calculate the p-value
  ## Calculate the chi-square for the random sample.
  observed_calculated_chisquare = calculate_chisquare(expected, observed, buckettype, buckets, axis)
  ## Get the p-value for this chi-square.
  p_value = sum(data['calculated_chi_square'] > observed_calculated_chisquare) / len(data)

  if p_value <= 0.001:
    print(color.BOLD +"Observed Chi-Square = " + color.END, observed_calculated_chisquare)
    print(color.BOLD +"Critical Chi-Square Value for 0.1% = " + color.END , critical_value_001_final)
    print(color.BOLD+'H0:'+color.END + 'The observed frequency distribution fits the expected frequency distribution.')
    print(color.BOLD+'P-value = '+color.END ,p_value, '***' )
    print(color.BOLD+'*** '+color.END + 'rejects null hypothesis for an alpha of 0.1%')
    print(color.BOLD+'**  '+color.END +'rejects null hypothesis for an alpha of 1%')
    print(color.BOLD+'*   ' + color.END+ 'rejects null hypothesis for an alpha of 5%')
  elif (p_value > 0.001 and p_value <= 0.01):
    print(color.BOLD +"Observed Chi-Square = "+ color.END , observed_calculated_chisquare)
    print(color.BOLD +"Critical Chi-Square Value for 1% = "+ color.END , critical_value_01_final)
    print(color.BOLD +'P-value = '+ color.END ,p_value, '**')
    print(color.BOLD +'H0:'+ color.END + 'The observed frequency distribution fits the expected frequency distribution.')
    print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
    print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
    print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')
  elif (p_value > 0.01 and p_value <= 0.05):
    print(color.BOLD +"Observed Chi-Square = "+ color.END , observed_calculated_chisquare)
    print(color.BOLD +"Critical Chi-Square Value for 5% = "+ color.END , critical_value_05_final)
    print(color.BOLD +'P-value = '+ color.END , p_value, '*' )
    print(color.BOLD +'H0:' + color.END + 'The observed frequency distribution fits the expected frequency distribution.')
    print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
    print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
    print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')
  else:
    print(color.BOLD +"Observed Chi-Square = "+ color.END , observed_calculated_chisquare)
    print(color.BOLD +'P-value = '+ color.END ,p_value)
    print(color.BOLD +'H0:' + color.END + 'The observed frequency distribution fits the expected frequency distribution.')
    print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
    print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
    print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')


