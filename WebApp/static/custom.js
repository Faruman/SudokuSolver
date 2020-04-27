function adjustSudokuBox() {
    $('.sudoku-box').each(function() {
            $(this).css('font-size', Math.max(0.3*parseFloat($(this).parent().css('width'))/9, 10));
            $(this).css('height', parseFloat($(this).parent().css('width'))/9)
        }
    )
}

$(document).ready(function() {
    adjustSudokuBox()
});

$( window ).resize(function() {
    adjustSudokuBox();
    //add function to add toggle classes to navbar header if screen size is below threshold
    if ($( window ).width() < 768){
        $('.navbar').each(function() {
            $(this).attr('data-toggle', "collapse");
            $(this).attr('data-target', "#navbarHeader")
        })
    } else {
        $('.navbar').each(function() {
            $(this).removeAttr('data-toggle');
            $(this).removeAttr('data-target')
        })
    }
});