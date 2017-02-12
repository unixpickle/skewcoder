package main

import (
	"fmt"
	"image/png"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/skewcoder"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "<net_file> <output.png> [features...]")
		os.Exit(1)
	}
	netFile := os.Args[1]
	outFile := os.Args[2]

	keepFeatures := map[int]bool{}
	for _, x := range os.Args[3:] {
		idx, err := strconv.Atoi(x)
		if err != nil {
			essentials.Die("invalid feature:", x)
		}
		keepFeatures[idx] = true
	}

	net := insertDropLayer(readNetwork(netFile), keepFeatures)

	img := mnist.ReconstructionGrid(func(img []float64) []float64 {
		imVec := anyvec32.MakeVectorData(anyvec32.MakeNumericList(img))
		v := anydiff.NewVar(imVec)
		res32 := net.Apply(v, 1).Output().Data().([]float32)
		res := make([]float64, len(res32))
		for i, x := range res32 {
			res[i] = float64(x)
		}
		return res
	}, mnist.LoadTestingDataSet(), 5, 5)

	f, _ := os.Create(outFile)
	defer f.Close()
	png.Encode(f, img)
}

func readNetwork(file string) anynet.Net {
	var net anynet.Net
	if err := serializer.LoadAny(file, &net); err != nil {
		essentials.Die("read net:", err)
	}
	return net
}

func insertDropLayer(net anynet.Net, keep map[int]bool) anynet.Net {
	if len(keep) == 0 {
		return net
	}
	dropLayer := &FeatureDropper{
		Keep:  keep,
		Means: meanFeatures(net),
	}
	for i, x := range net {
		if _, ok := x.(*skewcoder.Layer); ok {
			net = append(net, nil)
			copy(net[i+1:], net[i:])
			net[i] = dropLayer
			return net
		}
	}
	essentials.Die("No reconstructor layer")
	return nil
}

func meanFeatures(net anynet.Net) []float32 {
	samples := mnist.LoadTrainingDataSet()

	var featureLayers anynet.Net
	for i, x := range net {
		if _, ok := x.(*skewcoder.Layer); ok {
			featureLayers = net[:i]
			break
		}
	}

	var sum anyvec.Vector
	inputs := samples.IntensityVectors()
	for i := 0; i < 1000; i++ {
		sample := inputs[rand.Intn(len(inputs))]
		sampleVec := anyvec32.MakeVectorData(anyvec32.MakeNumericList(sample))
		fv := featureLayers.Apply(anydiff.NewConst(sampleVec), 1).Output()
		if sum == nil {
			sum = fv.Copy()
		} else {
			sum.Add(fv)
		}
	}
	sum.Scale(float32(1.0 / 1000))

	return sum.Data().([]float32)
}
