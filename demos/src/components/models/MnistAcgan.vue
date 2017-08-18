<template>
  <div class="demo mnist-acgan">
    <div class="title">
      <span>Auxiliary Classifier Generative Adversarial Networks (AC-GAN) on MNIST</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column is-half input-column">
        <div class="input-container">
          <canvas id="noise-canvas" width="120" height="120"></canvas>
          <div class="input-items">
            <div v-for="n in 10"
              class="digit-select" :class="{ active: digit === n - 1 }"
              @click="selectDigit(n - 1)"
            >{{ n - 1 }}</div>
            <div
              class="noise-btn"
              @click="onGenerateNewNoise"
            >Generate New Noise</div>
          </div>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <canvas id="output-canvas-scaled" width="180" height="180"></canvas>
          <canvas id="output-canvas" width="28" height="28" style="display:none;"></canvas>
        </div>
      </div>
      <div class="column controls-column">
        <mdl-switch v-model="useGpu" :disabled="modelLoading || !hasWebgl">use GPU</mdl-switch>
      </div>
    </div>
    <div class="layer-results-container"  v-if="!modelLoading" id="results-container">
    	<div id="webgl_container"></div>
    </div>
  </div>
</template>

<script>
import * as utils from '../../utils'
import filter from 'lodash/filter'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/mnist-acgan-arch'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/mnist_acgan/mnist_acgan.json',
  weights: '/demos/data/mnist_acgan/mnist_acgan_weights.buf',
  metadata: '/demos/data/mnist_acgan/mnist_acgan_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/mnist_acgan/mnist_acgan.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_acgan/mnist_acgan_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/mnist_acgan/mnist_acgan_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: this.hasWebgl,
      digit: 6,
      noiseVector: [],
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)), // eslint-disable-line
      modelLoading: true,
      output: new Float32Array(28 * 28),
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: []
    }
  },

  watch: {
    useGpu: function(value) {
      this.model.toggleGpu(value)
    }
  },

  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows: function() {
      const rows = []
      for (let row = 0; row < 12; row++) {
        rows.push(filter(this.architectureDiagram, { row }))
      }
      return rows
    }
  },

  created: function() {
    this.createNoise()
  },

  mounted: function() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.initWebgl()
        this.runModel()
        this.drawNoise()
      })
    })
  },

  methods: {
	initWebgl: function() {			
		const container = document.getElementById("webgl_container");
        this.container = container;

        this.originalWidth = 0.95*container.offsetWidth ;
		this.originalHeight = 1000;
        this.rotatingCam = false;
		this.posX = [];
		this.posY = [];
		this.posZ = [];
		this.layerNum = [];
		this.clicked = false;

		const scene = new THREE.Scene();
        //scene.background = new THREE.Color( 0xff0000 );
        this.scene = scene;

		scene.fog = new THREE.Fog( 0xffffff, 1, 30000 );
		scene.fog.color.setHSL( 0.6, 0, 1 );

		const highlightBox = new THREE.Mesh(
			new THREE.BoxGeometry( 12,12,12 ),
			new THREE.MeshLambertMaterial( { color: 0xffff00 }
		) );
		highlightBox.visible = false;
		scene.add( highlightBox );
        this.highlightBox = highlightBox;

	    const camera = new THREE.PerspectiveCamera( 75, this.originalWidth / this.originalHeight, 1, 25000 );
	    camera.position.z = 5000;
        this.camera = camera;

	    const renderer = new THREE.WebGLRenderer({ antialias: true});
        renderer.setClearColor( 0xFFAAFF );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.sortObjects = false;
	    renderer.setSize( this.originalWidth, this.originalHeight );
        this.renderer = renderer;

		var cameraControls = new THREE.OrbitControls( this.camera, renderer.domElement );
		cameraControls.target.set( 0, 0, 0 );

		var loader = new THREE.FontLoader();
		loader.load( 'https://raw.githubusercontent.com/rollup/three-jsnext/master/examples/fonts/droid/droid_sans_bold.typeface.json',this.setFont);

		this.raycaster = new THREE.Raycaster();
		this.mouse = new THREE.Vector2();

	    container.appendChild( renderer.domElement );
		renderer.domElement.addEventListener( 'mousemove', this.onMouseMove );
		renderer.domElement.addEventListener( 'mousedown', this.onMouseDown );
		renderer.domElement.addEventListener( 'click', this.onClick );
		renderer.domElement.addEventListener( 'mouseup', this.onMouseUp );
		window.addEventListener( 'resize', this.onWindowResize, false );
		this.animate();
	},

	animate: function() {
		requestAnimationFrame( this.animate );
		this.render();
    },
 
	render: function() {
		this.renderer.render( this.scene, this.camera );
	},

	setFont: function( font ) {
		this.font = font;
	},
		
	onWindowResize: function( e ) {
			//render();
	},
		
	onMouseDown: function( e ) {
		this.rotatingCam = true;
		this.clicked = true;
	},
	
	onClick: function( e ) {
		//console.log('click');
	},
	
	onMouseUp: function( e ) {
		//console.log('mouse up');
		this.rotatingCam = false;	
		this.clicked = false;
		
	},

	onMouseMove: function( e ) {
		event.preventDefault();
		this.mouse.x = ( e.clientX / window.innerWidth ) * 2 - 1;
		this.mouse.y = - ( e.clientY / window.innerHeight ) * 2 + 1;
	},
    selectDigit: function(digit) {
      this.digit = digit
      this.runModel()
    },
    createNoise: function() {
      const latentSize = 100
      const noiseVector = []
      for (let i = 0; i < latentSize; i++) {
        // uniform random between -1 and 1
        noiseVector.push(2 * Math.random() - 1)
      }
      this.noiseVector = noiseVector
    },
    runModel: function() {
      const inputData = {
        input_2: new Float32Array(this.noiseVector),
        input_3: new Float32Array([this.digit])
      }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['conv2d_7']
        this.drawOutput()
        this.getIntermediateResults()
      })
    },
    drawNoise: function() {
      // draw noise visualization on canvas
      const ctx = document.getElementById('noise-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      for (let x = 0; x < 10; x++) {
        for (let y = 0; y < 10; y++) {
          ctx.fillStyle = `rgba(57, 62, 70, ${(this.noiseVector[10 * x + y] + 1) / 2})`
          // scale 12x
          ctx.fillRect(12 * x, 12 * y, 12, 12)
        }
      }
    },
    drawOutput: function() {
      const ctx = document.getElementById('output-canvas').getContext('2d')
      const image = utils.image2Darray(this.output, 28, 28, [57, 62, 70])
      ctx.putImageData(image, 0, 0)

      // scale up
      const ctxScaled = document.getElementById('output-canvas-scaled').getContext('2d')
      ctxScaled.save()
      ctxScaled.scale(180 / 28, 180 / 28)
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      ctxScaled.drawImage(document.getElementById('output-canvas'), 0, 0)
      ctxScaled.restore()
    },
    onGenerateNewNoise: function() {
      this.createNoise()
      this.runModel()
      this.drawNoise()
    },
    getIntermediateResults: function() {
      let results = []
      for (let [name, layer] of this.model.modelLayersMap.entries()) {
        const layerClass = layer.layerClass || ''
        if (layerClass === 'InputLayer') continue

        let images = []
        if (layer.result && layer.result.tensor.shape.length === 3) {
          images = utils.unroll3Dtensor(layer.result.tensor)
        } else if (layer.result && layer.result.tensor.shape.length === 2) {
          images = [utils.image2Dtensor(layer.result.tensor)]
        } else if (layer.result && layer.result.tensor.shape.length === 1) {
          images = [utils.image1Dtensor(layer.result.tensor)]
        }
        results.push({ name, layerClass, images })
      }
      this.layerResultImages = results
      setTimeout(() => {
        this.displayOutput()
      }, 0)
    },
    clearIntermediateResults: function() {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor
        result.images.forEach((image, imageNum) => {
          const ctxScaled = document
            .getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`)
            .getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.restore()
        })
      })
    },
	displayOutput: function() {
        if(this.layerResultImages.length <= 1) {
			return;	
		}
		//if(this.points === null) { 
		var geometry = new THREE.Geometry();
		var colors = [];
		geometry.colors = colors;

		// material
		var material = new THREE.PointsMaterial( {
			size: 14,
			transparent: false,
			opacity: 0.9,
			vertexColors: THREE.VertexColors
		} );

		this.pointLayers = [];
		this.pointImages = [];
		this.pointMap = {};
		var nPoints = 0;
	    var layerX = 0; var layerY = 0; var layerZ = 0; var height = 0; var width = 0; var fc = false; var xPos = 0; var yPos = 0; var spacing = 0; var len = 0; 
		this.layerResultImages.forEach((result, layerNum) => {
	        var images = result.images;
	        len = Math.ceil(Math.sqrt(images.length));
	        layerZ = -layerNum * 1050;
	    	layerY =  0;
	        if(images.length==1 && images[0].height==1 || layerNum==0) {
	            fc = true;
	            spacing = 18;
			} else {
	            fc = false;
	            spacing = 10;
	    		xPos = layerY - len/2 *images[0].width*11;
				yPos = layerX + len/2 *images[0].height*11;
				var text = this.createLabel(result.name, 0, yPos+100, layerZ, 100, "white");
				this.scene.add(text);
			}
	    	images.forEach((image, imageNum) => {
	            width = image.width;
	            height = image.height;
	            if(fc) {  
					if(layerNum==0) {
			            len = Math.ceil(Math.sqrt(height));
			        	height = len;
			        	width = Math.ceil(height/width);
					} else {
			            len = Math.ceil(Math.sqrt(width));
			        	width = len;
			        	height = Math.ceil(width/height);
					}
	        		xPos = layerX - len/2 * 18;
	    			yPos = layerY + len/2 * 18;
					var text = this.createLabel(result.name, 0, yPos+100, layerZ, 30, "white");
					this.scene.add(text);
	            } else {
	        		xPos = layerY + (-len/2 + imageNum%len)*image.width*11;
	    			yPos = layerX - (-len/2 + Math.floor(imageNum/len))*image.height*11;
	            }
				for ( var xInd = 0; xInd < width; xInd ++ ) {
					for ( var yInd = 0; yInd < height; yInd ++ ) {
			            if(layerNum > 0 && fc && (xInd*width + yInd >= image.width) || layerNum==0 && fc && (xInd*width + yInd >= image.height)) { 
			            	break;
			            }
						var position = new THREE.Vector3();
						position.y = yPos-xInd*spacing;
						position.x = xPos+yInd*spacing;
						position.z = layerZ;

						this.pointMap[position] = nPoints;
						this.pointLayers.push(layerNum);
						this.pointImages.push(imageNum);
						// add it to the geometry
						geometry.vertices.push(position);
		
						var v = image.data[(xInd*width+yInd)*4];
						if(fc)
							v = image.data[(xInd*width+yInd)*4+3];
	                    colors.push(new THREE.Color( v/255.0,v/255.0,v/255.0));
	                    nPoints++;
					}
				}
			});
		});
		var pointCloud = new THREE.Points( geometry, material );
		this.scene.add( pointCloud );
		this.points = pointCloud;
		/*} else {
			var nPoints = 0;
			this.layerResultImages.forEach((result, layerNum) => {
		    	result.images.forEach((image, imageNum) => {
					for ( var xInd = 0; xInd < image.width; xInd ++ ) {
						for ( var yInd = 0; yInd < image.height; yInd ++ ) {
							var v = image.data[(xInd*width+yInd)*4+3];
		                    this.points.geometry.colors[nPoints].setRGB(v/255.0,v/255.0,v/255.0);
							nPoints++;
						}
					}
				});
			});
			this.points.geometry.colorsNeedUpdate = true;
		}*/		
	},
    createLabel: function(text, x, y, z, size, color) {
		var materialFront = new THREE.MeshBasicMaterial( { color: color} );
		var materialSide = new THREE.MeshBasicMaterial( { color: 0x000088 } );
		var materialArray = [ materialFront, materialSide ];
		var textGeom = new THREE.TextGeometry( text, 
		{
			size: size, height: 4, curveSegments: 3, font: this.font,
			bevelThickness: 1, bevelSize: 2, bevelEnabled: true,
			material: 0, extrudeMaterial: 1
		});
	
		var textMesh = new THREE.Mesh(textGeom, materialArray );
		textGeom.computeBoundingBox();
    	textGeom.textWidth = textGeom.boundingBox.max.x - textGeom.boundingBox.min.x;
		textMesh.position.set( x-textGeom.textWidth/2, y, z );
		return textMesh;
	}
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.mnist-acgan {
  & .column {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  & .column.input-column {
    justify-content: flex-end;

    & .input-container {
      margin: 20px;
      position: relative;
      user-select: none;
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;

      & canvas {
        margin: 10px;
      }

      & .input-items {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        width: 280px;

        & .digit-select, & .noise-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 50px;
          height: 50px;
          margin: 2px;
          border: 1px solid var(--color-green-lighter);
          color: var(--color-green);
          font-size: 20px;
          font-weight: bold;
          font-family: var(--font-monospace);
          transition: background-color 0.1s ease;
          cursor: default;

          &.active {
            background-color: var(--color-green);
            color: white;
          }

          &:hover:not(.active) {
            background-color: var(--color-green-lighter);
            cursor: pointer;
          }
        }

        & .noise-btn {
          width: 266px;
          font-size: 14px;
        }
      }

      & .canvas-container {
        position: relative;
        display: inline-flex;
        justify-content: flex-end;
        margin: 10px 0;
        border: 15px solid var(--color-green-lighter);
        transition: border-color 0.2s ease-in;

        & canvas {
          background: whitesmoke;
        }
      }
    }
  }

  & .column.output-column {
    & .output {
      border-radius: 10px;
      overflow: hidden;

      & canvas {
        background: whitesmoke;
      }
    }
  }

  & .column.controls-column {
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;

    & .mdl-switch {
      width: auto;
    }
  }

  & .architecture-container {
    min-width: 700px;
    max-width: 900px;
    margin: 0 auto;
    position: relative;

    & .layers-row {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      margin-bottom: 5px;
      position: relative;
      z-index: 1;

      & .layer-column {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 5px;

        & .layer {
          display: inline-block;
          background: whitesmoke;
          border: 2px solid var(--color-green);
          border-radius: 5px;
          padding: 2px 10px 0px;

          & .layer-class-name {
            color: var(--color-green);
            font-size: 14px;
            font-weight: bold;
          }

          & .layer-details {
            color: #999999;
            font-size: 12px;
            font-weight: bold;
          }
        }
      }
    }

    & .architecture-connections {
      position: absolute;
      top: 0;
      left: 0;
      z-index: 0;

      & path {
        stroke-width: 4px;
        stroke: #AAAAAA;
        fill: none;
      }
    }
  }
}
</style>
