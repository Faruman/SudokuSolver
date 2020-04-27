function adjustSudokuBox() {
    $('.sudoku-box').each(function() {
            $(this).css('font-size', 0.3*parseFloat($(this).parent().css('width'))/9);
            $(this).css('height', parseFloat($(this).parent().css('width'))/9)
        }
    )
}

function adjustButton(){
    $(".input-btn").each(function(){
        $(this).css('font-size', 0.3*parseFloat($('.sudoku').css('width'))/9)
    })
}

$(document).ready(function() {
    adjustSudokuBox()
    adjustButton()
});

$( window ).resize(function() {
    adjustSudokuBox()
    adjustButton()
    //add function to add toggle classes to navbar header if screen size is below threshold
});