import numpy as np
import pandas as pd

#bold print statements
class color:
    BOLD = '\033[1m'
    END = '\033[0m'

def calculate_psi(original, new, buckettype='bins', number=10):
    '''Calculate the PSI across all variables

    Args:
        original: numpy matrix of original values
        new: numpy matrix of new values, same size as expected
        number: enter the number of buckets

    Returns:
        psi_value: ndarray of PSI values for each variable
    '''

    def scaled(breakpoints, min, max):
        breakpoints += -(np.min(breakpoints))
        breakpoints /= np.max(breakpoints) / (max - min)
        breakpoints += min
        return breakpoints

    def sub_psi(og_perc, new_perc):
        '''
        Calculate the observed PSI value from comparing the values.
        Update the observed value to a very small number if equal to zero.
        '''
        if new_perc == 0:
            new_perc = 0.0001
        if og_perc == 0:
            og_perc = 0.0001
        subpsi = (og_perc - new_perc) * np.log(og_perc / new_perc)
        return (subpsi)

    def psi_finder(original, new, buckettype='bins', number=10):
        raw_breakp = np.arange(0, number + 1) / (number) * 100

        if buckettype == 'bins':
            breakpoints = scaled(raw_breakp, np.min(original), np.max(original))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(original, b) for b in raw_breakp])

        og_counts = np.histogram(original, breakpoints)[0]
        new_counts = np.histogram(new, breakpoints)[0]

        og_perc = og_counts / len(original)
        new_perc = new_counts / len(new)

        psi_value = 0
        for i in range(0, len(og_perc)):
            psi_value += sub_psi(og_perc[i], new_perc[i])

        return (psi_value)

    data = pd.DataFrame([])

    for i in np.arange(1000):  # generates 1000 samples
        bs_sample = np.random.choice(original, len(original))  # creating bootstrap sample
        calculated_psi = psi_finder(original, bs_sample)
        data = pd.concat([data, pd.DataFrame({'calculated_psi': calculated_psi}, index=[0])], ignore_index=True)

    critical_value_05 = data.quantile(0.95)
    critical_value_01 = data.quantile(0.99)
    critical_value_001 = data.quantile(0.999)

    # Calculate the PSI for the random sample, generated at the beginning.
    psi_val = psi_finder(original, new)
    # Get the p-value for this PSI.
    p_value = sum(data['calculated_psi'] > psi_val) / len(data)

    if p_value <= 0.001:
        print(color.BOLD + "Observed PSI = " + color.END, psi_val)
        print(color.BOLD + "Critical PSI Value for 0.1% = " + color.END, critical_value_001['calculated_psi'], '\n')
        print(color.BOLD + "P-value = " + color.END, p_value, "***")
        print(color.BOLD + "H0:" + color.END + " Both samples are equal.")
        print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
        print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
        print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')
    elif (p_value > 0.001 and p_value <= 0.01):
        print(color.BOLD + "Observed PSI = " + color.END, psi_val)
        print(color.BOLD + "Critical PSI Value for 1% = " + color.END, critical_value_01['calculated_psi'])
        print(color.BOLD + "P-value = " + color.END, p_value, "**")
        print(color.BOLD + "H0:" + color.END + " Both samples are equal.")
        print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
        print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
        print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')
    elif (p_value > 0.01 and p_value <= 0.05):
        print(color.BOLD + "Observed PSI = " + color.END, psi_val)
        print(color.BOLD + "Critical PSI Value for 5% = " + color.END, critical_value_05['calculated_psi'])
        print(color.BOLD + "P-value = " + color.END, p_value, "*")
        print(color.BOLD + "H0:" + color.END + " Both samples are equal.")
        print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
        print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
        print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')
    else:
        print(color.BOLD + "Observed PSI = " + color.END, psi_val)
        print(color.BOLD + "P-value = " + color.END, p_value)
        print(color.BOLD + "H0:" + color.END + " Both samples are equal.")
        print(color.BOLD + '*** ' + color.END + 'rejects null hypothesis for an alpha of 0.1%')
        print(color.BOLD + '**  ' + color.END + 'rejects null hypothesis for an alpha of 1%')
        print(color.BOLD + '*   ' + color.END + 'rejects null hypothesis for an alpha of 5%')

    return psi_val
