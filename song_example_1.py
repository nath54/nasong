#
### Import Modules. ###
#
import lib_value as lv


#
### Specify duration (in seconds). ###
#
duration: float = 20.0

#
### Song description, in function of Time (in seconds). ###
#
def song(time: lv.Value) -> lv.Value:

    #
    a = lv.Sin(
        value=time,
        frequency=lv.Constant(640),
        amplitude=lv.Constant(0.8)
    )

    #
    aaa = lv.Sin(
        value=time,
        frequency=lv.Constant(10),
        amplitude=lv.Constant(0.8)
    )

    #
    aa = lv.Sin(
        value=time,
        frequency=lv.BasicScaling(value=aaa, mult_scale=lv.Constant(100), sum_scale=lv.Constant(200)),
        amplitude=lv.Constant(0.1)
    )

    #
    b = lv.RandomFloat(
        min_range=lv.Constant(-0.05),
        max_range=lv.Constant(0.1)
    )

    #
    c = lv.Sum(a, aa, b)

    #
    return c

