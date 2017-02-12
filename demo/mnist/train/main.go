package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/skewcoder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var netPath string
	var stepSize float64
	var batchSize int

	flag.StringVar(&netPath, "out", "out_net", "out net path")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&batchSize, "batch", 16, "SGD batch size")

	flag.Parse()

	var net anynet.Net
	if err := serializer.LoadAny(netPath, &net); err != nil {
		log.Println("Creating network...")
		net = createNetwork()
	} else {
		log.Println("Loaded network.")
	}

	trainer := &anyff.Trainer{
		Net:     net,
		Cost:    anynet.MSE{},
		Params:  net.Parameters(),
		Average: true,
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     samples(mnist.LoadTrainingDataSet()),
		Rater:       anysgd.ConstRater(stepSize),
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())

	if err := serializer.SaveAny(netPath, net); err != nil {
		essentials.Die("save network:", err)
	}
}

func createNetwork() anynet.Net {
	c := anyvec32.CurrentCreator()
	inMarkup := `
		Input(w=28, h=28, d=1)
		Conv(w=5, h=5, n=16, sx=2, sy=2)
		Tanh
		Conv(w=3, h=3, n=16, sx=2, sy=2)
		Tanh
		FC(out=128)
		Tanh
		FC(out=30)
		Tanh
	`
	inConv, err := anyconv.FromMarkup(c, inMarkup)
	if err != nil {
		essentials.Die("create network:", err)
	}

	outMarkup := `
		Input(w=4, h=4, d=8)
		Padding(t=1, r=1, b=1, l=1)
		Conv(w=3, h=3, n=8)
		Tanh
		Resize(w=16, h=16)
		Conv(w=3, h=3, n=8)
		Tanh
		Resize(w=30, h=30)
		Conv(w=3, h=3, n=1)
		Sigmoid
	`
	outConv, err := anyconv.FromMarkup(c, outMarkup)
	if err != nil {
		essentials.Die("create network:", err)
	}

	skew := &skewcoder.Layer{
		Block: anyrnn.NewLSTM(c, 1, 128).ScaleInWeights(c.MakeNumeric(5)),
	}

	return append(append(inConv.(anynet.Net), skew), outConv.(anynet.Net)...)
}

func samples(d mnist.DataSet) anysgd.SampleList {
	var res anyff.SliceSampleList
	for _, x := range d.IntensityVectors() {
		c := anyvec32.CurrentCreator()
		v := c.MakeVectorData(c.MakeNumericList(x))
		res = append(res, &anyff.Sample{
			Input:  v,
			Output: v.Copy(),
		})
	}
	return res
}
