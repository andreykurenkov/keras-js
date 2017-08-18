<template>
  <div class="demo mnist-vae">
    <div class="title">
      <span>Convolutional Variational Autoencoder, trained on MNIST</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column is-3 controls-column">
        <mdl-switch v-model="useGpu" :disabled="modelLoading || !hasWebgl">use GPU</mdl-switch>
        <div class="coordinates">
          <div class="coordinates-x">x: {{ inputCoordinates[0] < 0 ? inputCoordinates[0].toFixed(2) : inputCoordinates[0].toFixed(3) }}</div>
          <div class="coordinates-y">y: {{ inputCoordinates[1] < 0 ? inputCoordinates[1].toFixed(2) : inputCoordinates[1].toFixed(3) }}</div>
        </div>
      </div>
      <div class="column input-column">
        <div class="input-container">
          <div class="input-label">Click around the latent space <span class="arrow">⤸</span></div>
          <div class="canvas-container">
            <canvas
              id="input-canvas" width="200" height="200"
              @mouseenter="activateCrosshairs"
              @mouseleave="deactivateCrosshairs"
              @mousemove="draw"
              @click="selectCoordinates"
              @touchend="selectCoordinates"
            ></canvas>
            <div class="axis x-axis">
              <span>-1</span>
              <span>x</span>
              <span>1</span>
            </div>
            <div class="axis y-axis">
              <span>-1</span>
              <span>y</span>
              <span>1</span>
            </div>
          </div>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <canvas id="output-canvas-scaled" width="180" height="180"></canvas>
          <canvas id="output-canvas" width="28" height="28" style="display:none;"></canvas>
        </div>
      </div>
    </div>
    <div class="layer-results-container"  v-if="!modelLoading" id="results-container">
    	<div id="webgl_container"></div>
    </div>
    <div class="layer-results-container" v-if="!modelLoading">
      <div class="bg-line"></div>
      <div
        v-for="(layerResult, layerIndex)  in layerResultImages"
        :key="`intermediate-result-${layerIndex}`"
        class="layer-result"
      >
        <div class="layer-result-heading">
          <span class="layer-class">{{ layerResult.layerClass }}</span>
          <span> {{ layerDisplayConfig[layerResult.name].heading }}</span>
        </div>
        <div class="layer-result-canvas-container">
          <canvas v-for="(image, index) in layerResult.images"
            :key="`intermediate-result-${layerIndex}-${index}`"
            :id="`intermediate-result-${layerIndex}-${index}`"
            :width="image.width"
            :height="image.height"
            style="display:none;"
          ></canvas>
          <canvas v-for="(image, index) in layerResult.images"
            :key="`intermediate-result-${layerIndex}-${index}-scaled`"
            :id="`intermediate-result-${layerIndex}-${index}-scaled`"
            :width="layerDisplayConfig[layerResult.name].scalingFactor * image.width"
            :height="layerDisplayConfig[layerResult.name].scalingFactor * image.height"
          ></canvas>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as utils from '../../utils'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/mnist_vae/mnist_vae.json',
  weights: '/demos/data/mnist_vae/mnist_vae_weights.buf',
  metadata: '/demos/data/mnist_vae/mnist_vae_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/mnist_vae/mnist_vae.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_vae/mnist_vae_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/mnist_vae/mnist_vae_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

const LAYER_DISPLAY_CONFIG = {
  dense_19: { heading: 'input dimensions = 2, output dimensions = 128, ReLU activation', scalingFactor: 2 },
  dense_20: { heading: 'ReLU activation, output dimensions = 25088 (64 x 14 x 14)', scalingFactor: 2 },
  reshape_7: { heading: '', scalingFactor: 2 },
  conv2d_transpose_19: { heading: '64 3x3 filters, padding same, 1x1 strides, ReLU activation', scalingFactor: 2 },
  conv2d_transpose_20: { heading: '64 3x3 filters, padding same, 1x1 strides, ReLU activation', scalingFactor: 2 },
  conv2d_transpose_21: { heading: '64 2x2 filters, padding valid, 2x2 strides, ReLU activation', scalingFactor: 2 },
  conv2d_15: { heading: '1 2x2 filters, padding same, 1x1 strides, sigmoid activation', scalingFactor: 2 }
}

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: this.hasWebgl,
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)), // eslint-disable-line
      modelLoading: true,
      output: new Float32Array(28 * 28),
      crosshairsActivated: false,
      inputCoordinates: [-0.3, -0.6],
      position: [35, 20],
      layerResultImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG
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
    }
  },

  mounted: function() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.drawPosition()
        this.initWebgl()
        this.getIntermediateResults()
        this.runModel()
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

		scene.fog = new THREE.Fog( 0xffffff, 1, 25000 );
		scene.fog.color.setHSL( 0.6, 0, 1 );

		const highlightBox = new THREE.Mesh(
			new THREE.BoxGeometry( 12,12,12 ),
			new THREE.MeshLambertMaterial( { color: 0xffff00 }
		) );
		highlightBox.visible = false;
		scene.add( highlightBox );
        this.highlightBox = highlightBox;

	    const camera = new THREE.PerspectiveCamera( 75, this.originalWidth / this.originalHeight, 1, 15000 );
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
    activateCrosshairs: function() {
      this.crosshairsActivated = true
    },
    deactivateCrosshairs: function(e) {
      this.crosshairsActivated = false
      this.draw(e)
    },
    draw: function(e) {
      const [x, y] = this.getEventCanvasCoordinates(e)
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, 200, 200)

      this.drawPosition()

      if (this.crosshairsActivated) {
        ctx.strokeStyle = '#1BBC9B'
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, 200)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(200, y)
        ctx.stroke()
      }
    },
    drawPosition: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, 200, 200)
      ctx.fillStyle = 'rgb(57, 62, 70)'
      ctx.beginPath()
      ctx.arc(...this.position, 5, 0, Math.PI * 2, true)
      ctx.closePath()
      ctx.fill()
    },
    getEventCanvasCoordinates: function(e) {
      let { clientX, clientY } = e
      // for touch event
      if (e.touches && e.touches.length) {
        clientX = e.touches[0].clientX
        clientY = e.touches[0].clientY
      }

      const canvas = document.getElementById('input-canvas')
      const { left, top } = canvas.getBoundingClientRect()
      const [x, y] = [clientX - left, clientY - top]
      return [x, y]
    },
    selectCoordinates: function(e) {
      const [x, y] = this.getEventCanvasCoordinates(e)
      if (!this.model.isRunning) {
        this.position = [x, y]
        this.inputCoordinates = [x * 2 / 200 - 1, y * 2 / 200 - 1]
        this.draw(e)
        this.runModel()
      }
    },
    runModel: function() {
      const inputData = { input_7: new Float32Array(this.inputCoordinates) }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['conv2d_15']
        this.drawOutput()
        this.getIntermediateResults()
      })
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
        this.showIntermediateResults()
        this.displayOutput()
      }, 0)
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
	        layerZ = -layerNum * 650;
	    	layerY =  0;
	        if(images.length==1 && images[0].height==1) {
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
	                len = Math.ceil(Math.sqrt(width));
	            	width = len;
	            	height = Math.ceil(width/height);
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
			            if(fc && (xInd*width + yInd >= image.width)) { 
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
		this.textPivots = [];
		for(var i = 0 ; i<10; i++){
			var text = this.createLabel(i.toString(), (i-4)*100, yPos+2000, layerZ+200*this.output[i], 600*this.output[i], "cyan");
			this.scene.add(text);
			this.textPivots.push(text);
		}
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
	},	
    showIntermediateResults: function() {
      this.layerResultImages.forEach((result, layerNum) => {
        const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor
        result.images.forEach((image, imageNum) => {
          const ctx = document.getElementById(`intermediate-result-${layerNum}-${imageNum}`).getContext('2d')
          ctx.putImageData(image, 0, 0)
          const ctxScaled = document
            .getElementById(`intermediate-result-${layerNum}-${imageNum}-scaled`)
            .getContext('2d')
          ctxScaled.save()
          ctxScaled.scale(scalingFactor, scalingFactor)
          ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
          ctxScaled.drawImage(document.getElementById(`intermediate-result-${layerNum}-${imageNum}`), 0, 0)
          ctxScaled.restore()
        })
      })
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
    }
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.mnist-vae {
  & .column {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  & .column.controls-column {
    flex-direction: column;
    align-items: flex-end;
    justify-content: flex-start;
    padding-top: 80px;

    & .mdl-switch {
      width: auto;
      margin-right: 15px;
    }

    & .coordinates {
      margin-left: 5px;
      margin-top: 45px;
      font-family: var(--font-monospace);
      font-size: 20px;
      color: var(--color-lightgray);
    }
  }

  & .column.input-column {
    justify-content: center;

    & .input-container {
      text-align: right;
      margin: 20px;
      position: relative;
      user-select: none;

      & .input-label {
        font-family: var(--font-cursive);
        font-size: 18px;
        color: var(--color-lightgray);
        text-align: right;

        & span.arrow {
          font-size: 36px;
          color: #CCCCCC;
          position: absolute;
          right: -32px;
          top: 8px;
        }
      }

      & .canvas-container {
        position: relative;
        display: inline-flex;
        justify-content: flex-end;
        margin: 10px 0;
        border: 15px solid var(--color-green-lighter);
        transition: border-color 0.2s ease-in;

        &:hover {
          border-color: var(--color-green-light);
        }

        & canvas {
          background: whitesmoke;

          &:hover {
            cursor: crosshair;
          }
        }

        & .axis {
          position: absolute;
          cursor: default;
          user-select: none;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-family: var(--font-monospace);
          font-size: 14px;
          color: var(--color-green);
        }

        & .axis.x-axis {
          right: 0;
          bottom: -45px;
          width: 200px;
          flex-direction: row;
        }

        & .axis.y-axis {
          top: 0;
          left: -55px;
          height: 200px;
          flex-direction: column;
        }
      }
    }
  }

  & .column.output-column {
    justify-content: flex-start;

    & .output {
      border-radius: 10px;
      overflow: hidden;

      & canvas {
        background: whitesmoke;
      }
    }
  }

  & .layer-results-container {
    position: relative;

    & .bg-line {
      position: absolute;
      z-index: 0;
      top: 0;
      left: 50%;
      background: whitesmoke;
      width: 15px;
      height: 100%;
    }

    & .layer-result {
      position: relative;
      z-index: 1;
      margin: 30px 20px;
      background: whitesmoke;
      border-radius: 10px;
      padding: 20px;
      overflow-x: auto;

      & .layer-result-heading {
        font-size: 1rem;
        color: #999999;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        font-size: 12px;

        & span.layer-class {
          color: var(--color-green);
          font-size: 14px;
          font-weight: bold;
        }
      }

      & .layer-result-canvas-container {
        display: inline-flex;
        flex-wrap: wrap;
        background: whitesmoke;

        & canvas {
          border: 1px solid lightgray;
          margin: 1px;
        }
      }
    }
  }
}
</style>
