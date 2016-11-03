package main

import (
	"fmt"
	"image/png"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/skewcoder"
	"github.com/unixpickle/weakai/neuralnet"
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
			die("Invalid feature:", x)
		}
		keepFeatures[idx] = true
	}

	net := insertDropLayer(readNetwork(netFile), keepFeatures)

	img := mnist.ReconstructionGrid(func(img []float64) []float64 {
		v := &autofunc.Variable{Vector: img}
		return net.Apply(v).Output()
	}, mnist.LoadTestingDataSet(), 5, 5)

	f, _ := os.Create(outFile)
	defer f.Close()
	png.Encode(f, img)
}

func readNetwork(file string) neuralnet.Network {
	netData, err := ioutil.ReadFile(file)
	if err != nil {
		die("Read network:", err)
	}
	net, err := neuralnet.DeserializeNetwork(netData)
	if err != nil {
		die("Deserialize network:", err)
	}
	return net
}

func insertDropLayer(net neuralnet.Network, keep map[int]bool) neuralnet.Network {
	if len(keep) == 0 {
		return net
	}
	dropLayer := &FeatureDropper{Keep: keep}
	for i, x := range net {
		if _, ok := x.(*skewcoder.Reconstructor); ok {
			net = append(net, nil)
			copy(net[i+1:], net[i:])
			net[i] = dropLayer
			return net
		}
	}
	die("No reconstructor layer")
	return nil
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
