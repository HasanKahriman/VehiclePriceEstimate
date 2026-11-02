import numpy as np
from  scipy import stats

def transmission_price_difference_test(cursor,alpha=0.05):
    cursor.execute("SELECT price,transmission FROM CarListings")
    rows=cursor.fetchall()

    if not rows:
        raise ValueError("No data found!")

    automatic_price=np.array([price for price, transmission in rows if transmission == "Automatic"])
    manual_price=np.array([price for price, transmission in rows if transmission == "Manual"])

    automatic_mean=automatic_price.mean()
    manual_mean=manual_price.mean()

    _, p_levene = stats.levene(automatic_price, manual_price)
    has_equal_var = p_levene >= alpha
    t_stat, p_value = stats.ttest_ind(automatic_price, manual_price, equal_var=has_equal_var)

    h0_rejected = bool(p_value < alpha)
    print("----- HYPOTHESIS TEST RESULTS -----")
    print(f"H0: Price means are the same")
    print(f"H1: Price means are different")
    if h0_rejected:
        comment="Reject null hypothesis.\nThere is a difference between the means of prices automatic and manual vehicles."
    else:
        comment="Fail to reject null hypothesis.\nThere is no difference between the means of prices automatic and manual vehicles."

    return {
        "Automatic vehicles mean": round(automatic_mean,2),
        "Manual vehicles mean":round(manual_mean,2),
        "t-statistic": round(t_stat,4),
        "p-value": p_value,
        "p_levene": p_levene,
        "h0_rejected":h0_rejected,
        "Comment":comment
    }

def price_damage_record_difference_test(cursor,alpha=0.05):
    cursor.execute("SELECT price,damage_record FROM CarListings")
    rows=cursor.fetchall()

    if not rows:
        raise ValueError("No data found!")

    damage_record_price=np.array([price for price, damage_record in rows if damage_record > 0])
    without_damage_record_price=np.array([price for price, damage_record in rows if damage_record == 0])

    damage_record_mean=damage_record_price.mean()
    without_damage_record_mean=without_damage_record_price.mean()

    _, p_levene=stats.levene(damage_record_price, without_damage_record_price)
    has_equal_var = p_levene >= alpha
    t_stat, p_value=stats.ttest_ind(damage_record_price, without_damage_record_price, equal_var=has_equal_var)

    h0_rejected = bool(p_value < alpha)
    print("----- HYPOTHESIS TEST RESULTS -----")
    print(f"H0: Price means are the same")
    print(f"H1: Price means are different")
    if h0_rejected:
        comment="Reject null hypothesis.\nThere is a difference between the means of prices vehicles with and without damage records."
    else:
        comment="Fail to reject null hypothesis.\nThere is no difference between the means of prices vehicles with and without damage records."

    return {
        "Vehicles with damage record mean": round(damage_record_mean, 2),
        "Vehicles without damage record mean": round(without_damage_record_mean, 2),
        "t-statistic": round(t_stat, 4),
        "p-value": p_value,
        "p_levene": p_levene,
        "h0_rejected": h0_rejected,
        "Comment": comment
    }

def age_and_price_difference_test(cursor,alpha=0.05):
    cursor.execute("SELECT age,price FROM CarListings")
    rows=cursor.fetchall()

    if not rows:
        raise ValueError("No data found!")

    ages=np.array([age for age , price in rows])
    prices=np.array([price for age , price in rows])

    corr_coeff,p_value=stats.pearsonr(ages,prices)

    direction="negative" if corr_coeff < 0 else "positive"
    abs_r=abs(corr_coeff)

    if abs_r < 0.3:
        strength="weak"
    elif abs_r < 0.7:
        strength="moderate"
    else:
        strength="strong"

    h0_rejected=bool(p_value < alpha)

    print("----- HYPOTHESIS TEST RESULTS -----")
    print("H0: There is no correlation between vehicle age and price (r = 0)")
    print("H1: There is a correlation between vehicle age and price (r != 0)")

    if h0_rejected:
        comment=f"Reject null hypothesis.\nThere is a statistically significant {strength} {direction} correlation between vehicle age and price."
    else:
        comment="Fail to reject null hypothesis.\nThere is no statistically significant correlation between vehicle age and price."

    return {
        "Pearson correlation": round(corr_coeff, 4),
        "Correlation strength": strength.capitalize(),
        "Correlation direction": direction.capitalize(),
        "p-value": p_value,
        "h0_rejected": h0_rejected,
        "Comment": comment
    }




