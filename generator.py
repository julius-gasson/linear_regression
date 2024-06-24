import sys
from linear_regression import get_weights
import argparse

# input Valore: Float64
# trigger Valore < 0.022 "Pressure too low"
# trigger 0.04 < Valore ∧ Valore < 150.0 "Pressure too high"

default_weights = [-0.04225337851094229, 0.10247139873175348, -0.07277887407175535,
                -0.018073129781422195, 0.06274628142025422, -0.007973749459843495,
                -0.0016075391619728135, 0.041326514583673195, 0.984542071869799,
                -0.013424849503789566, 0.04221882327093451, 0.02864976398609131,
                -0.008972700489565206, 0.06609032404734135, -0.04206854767603739,
                -0.03254346565372635, 0.03704798244122886, -0.004308398260728902,
                -0.038037580170373786, 0.03718932087243454, -0.010246263061479098,
                -0.024012415901640865, 0.004424835919865436, 0.053332605803590116,
                -0.003624252549848439, 0.015462530939931864]

default_bias = -0.003625678496476388

def generate_spec(weights, bias, outfile):

    ### OPEN THE OUTPUT FILE ###
    try:
        of = open(outfile, "w")
    except:
        print("Error opening output file")
        sys.exit(1)

    ### WRITE THE SPECIFICATION ###
    for i in range(1, 28):
        of.write(f"input P{i}: Float64\n")
    of.write("output expected: Float64 := ")
    for i in range(2, 27):
        of.write(f"P{i} * {weights[i-1]}\n + ")
    of.write(f"{bias}\n")
    of.write("output divergence: Float64 := P1 - expected\n")
    of.write("trigger divergence < -0.0001 \"Pressure lower than expected\"\n")
    of.write("trigger divergence > 0.0001 \"Pressure higher than expected\"\n")

    of.close()


def main():
    parser = argparse.ArgumentParser(description="Generate specifications for pressure system")
    parser.add_argument('-i', '--infile', type=str, nargs='?', default='',
                        help='Input file for weights or "default" to use default values')
    parser.add_argument('-o', '--outfile', type=str, default="spec.lola", help='Output file to write the specifications')
    
    args = parser.parse_args()

    if args.infile == "":
        weights, bias = default_weights, default_bias
    else:
        weights, bias = get_weights(args.infile)
    
    generate_spec(weights, bias, args.outfile)

if __name__ == '__main__':
    main()